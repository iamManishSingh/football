/** @import { BBOX } from "../../utils/math.js" */

/**
 * ⚽ Advanced Ball Detection Post-Processing for RF-DETR
 * 
 * Drop-in replacement for ball-filter-rfdetr.js
 * Same interface: ballFilterStream() → TransformStream
 * Same data contract: data.ball = { bboxes, confidences, classIds }
 * 
 * Addresses:
 *  1) False positives (shoes, round objects) — size/aspect/shoe/grass filters
 *  2) Aerial ball lost & not re-acquired — Kalman prediction + grace period + search zone
 *  3) Outside pitch detections — grass-based ROI + frame edge exclusion
 *  4) Fast ball out of sync — Kalman smoothing + velocity-aware search radius
 */

// ============================================================================
// CONFIG — All tunable parameters in one place
// ============================================================================

export const FILTER_CONFIG = {
  // ── Geometry Filters ──
  // Rejects shoes (too large/elongated) and noise (too small)
  min_bbox_area: 12,                 // Min bbox area in px² (very small ball at distance)
  max_bbox_area: 7000,               // Max bbox area in px² (anything bigger is not a ball)
  aspect_ratio_min: 0.45,            // width/height — ball is roughly square
  aspect_ratio_max: 2.2,             // shoes/limbs are elongated, reject them

  // ── Frame Edge Exclusion ──
  // Rejects detections in broadcast bars, scoreboards, crowd
  exclude_top_ratio: 0.0,            // Exclude top X% of frame (increase for crowd/sky)
  exclude_bottom_pixels: 15,         // Exclude bottom N pixels (broadcast bar)
  exclude_left_ratio: 0.0,           // Exclude left edge
  exclude_right_ratio: 0.0,          // Exclude right edge

  // ── Grass / Pitch Verification ──
  // Ball in the lower part of frame should be near green pitch
  grass_check_zone_ratio: 0.55,      // Only check grass for detections below this % of frame height
  grass_threshold: 0.10,             // Min fraction of surrounding pixels that must be green
  grass_sample_pad: 8,               // Pixels to sample around bbox for grass check

  // HSV ranges for grass detection
  grass_hue_min: 0.22,
  grass_hue_max: 0.47,
  grass_sat_min: 0.15,
  grass_val_min: 0.15,

  // ── Shoe / Player Proximity Filter ──
  // Detections near player feet are likely shoes
  shoe_proximity_ratio: 0.45,        // Detection within this × player_width of foot center → suspect
  shoe_foot_zone_ratio: 0.85,        // Player foot zone starts at this % of player height
  shoe_score_threshold: 1.2,         // Accumulated shoe score to trigger suppression
  shoe_lock_frames: 3,               // Consecutive shoe-like frames before locking out
  shoe_bottom_zone_ratio: 0.78,      // Extra shoe penalty when ball is in bottom 78% area of player

  // ── Confidence Thresholds (Two-Tier) ──
  // Global: strict to reject false positives
  // Search zone: lenient to re-acquire ball after aerial play
  confidence_global: 0.22,           // Strict threshold outside search zone
  confidence_search_zone: 0.15,      // Lenient threshold inside predicted search zone

  // ── Kalman Filter ──
  // Physics-based tracker: predicts trajectory during gaps (aerial play, fast movement)
  kalman_process_noise: 6.0,         // Higher = more responsive to changes, lower = smoother
  kalman_measurement_noise: 2.0,     // Higher = trusts prediction more, lower = follows detection
  kalman_gravity: 0.4,               // Vertical acceleration in px/frame² (for aerial trajectories)

  // ── Track State Machine ──
  max_missing_frames: 25,            // Grace period: keep predicting for N frames without detection
  confidence_decay: 0.88,            // Multiply confidence by this each predicted frame
  min_predicted_confidence: 0.03,    // Stop predicting when confidence drops below this
  tentative_detections: 2,           // Need N detections in M frames to confirm track
  tentative_window: 5,               // Window of M frames for track confirmation

  // ── Motion / Temporal Consistency ──
  max_motion_px: 300,                // Max allowed jump between consecutive frames (pixels)
  search_radius_multiplier: 3.0,     // Search radius = speed × this
  min_search_radius: 50,             // Minimum search zone radius (pixels)
  max_search_radius: 350,            // Maximum search zone radius (pixels)

  // ── Scene Cut Detection ──
  scene_cut_time_gap: 1.0,           // Reset tracker if time gap > this (seconds)

  // ── Debug ──
  debug: false,
}


// ============================================================================
// KALMAN FILTER — 6-state: [x, y, vx, vy, ax, ay]
// ============================================================================
// Tracks position, velocity, AND acceleration.
// The acceleration state is what makes aerial ball tracking work —
// gravity pulls ay downward, creating a parabolic prediction.

class KalmanFilter {
  constructor(processNoise, measurementNoise, gravity) {
    this.gravity = gravity
    this.state = new Float64Array(6)  // [x, y, vx, vy, ax, ay]
    this.initialized = false

    // State covariance (6×6) — start with high uncertainty
    this.P = this._eye(6, 500)

    // Transition matrix: constant acceleration, dt = 1 frame
    //   x' = x + vx + 0.5*ax
    //   vx' = vx + ax
    //   ax' = ax  (acceleration changes slowly)
    this.F = [
      [1, 0, 1, 0, 0.5, 0  ],
      [0, 1, 0, 1, 0,   0.5],
      [0, 0, 1, 0, 1,   0  ],
      [0, 0, 0, 1, 0,   1  ],
      [0, 0, 0, 0, 1,   0  ],
      [0, 0, 0, 0, 0,   1  ],
    ]

    // Measurement: we only observe [x, y]
    this.H = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
    ]

    // Process noise — less noise on acceleration (changes slowly)
    this.Q = this._eye(6, processNoise)
    this.Q[4][4] = processNoise * 0.1
    this.Q[5][5] = processNoise * 0.1

    // Measurement noise
    this.R = this._eye(2, measurementNoise)
  }

  initialize(x, y) {
    this.state[0] = x
    this.state[1] = y
    this.state[2] = 0   // vx
    this.state[3] = 0   // vy
    this.state[4] = 0   // ax
    this.state[5] = this.gravity  // ay — assume gravity from start
    this.P = this._eye(6, 100)
    this.initialized = true
  }

  /** Predict next state. Call ONCE per frame before update(). Returns [x, y]. */
  predict() {
    if (!this.initialized) return [0, 0]

    const s = this.state
    const ns = new Float64Array(6)
    for (let i = 0; i < 6; i++) {
      let sum = 0
      for (let j = 0; j < 6; j++) sum += this.F[i][j] * s[j]
      ns[i] = sum
    }
    this.state = ns

    // P = F·P·Fᵀ + Q
    this.P = this._addMat(this._FPFt(this.F, this.P), this.Q)

    return [this.state[0], this.state[1]]
  }

  /** Update with measurement [x, y]. Call after predict(). Returns corrected [x, y]. */
  update(x, y) {
    if (!this.initialized) { this.initialize(x, y); return [x, y] }

    // Innovation
    const innov = [x - this.state[0], y - this.state[1]]

    // S = H·P·Hᵀ + R  (2×2)
    const S = [
      [this.P[0][0] + this.R[0][0], this.P[0][1] + this.R[0][1]],
      [this.P[1][0] + this.R[1][0], this.P[1][1] + this.R[1][1]],
    ]

    // S⁻¹
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
    const id = Math.abs(det) < 1e-12 ? 1e8 : 1 / det
    const Si = [
      [ S[1][1] * id, -S[0][1] * id],
      [-S[1][0] * id,  S[0][0] * id],
    ]

    // K = P·Hᵀ·S⁻¹  (6×2)
    const K = []
    for (let i = 0; i < 6; i++) {
      const ph0 = this.P[i][0], ph1 = this.P[i][1]
      K[i] = [
        ph0 * Si[0][0] + ph1 * Si[1][0],
        ph0 * Si[0][1] + ph1 * Si[1][1],
      ]
    }

    // state += K · innovation
    for (let i = 0; i < 6; i++) {
      this.state[i] += K[i][0] * innov[0] + K[i][1] * innov[1]
    }

    // P = (I - K·H)·P
    const KH = this._eye(6, 0)
    for (let i = 0; i < 6; i++) {
      KH[i][0] = K[i][0]
      KH[i][1] = K[i][1]
    }
    const IKH = this._eye(6, 1)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++)
        IKH[i][j] -= KH[i][j]

    const newP = this._eye(6, 0)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++)
        for (let k = 0; k < 6; k++)
          newP[i][j] += IKH[i][k] * this.P[k][j]
    this.P = newP

    return [this.state[0], this.state[1]]
  }

  getVelocity()  { return [this.state[2], this.state[3]] }

  getSpeed() { return Math.hypot(this.state[2], this.state[3]) }

  // ── Compact matrix helpers (fixed 6×6, no allocations on hot path) ──

  _eye(n, s) {
    const m = []
    for (let i = 0; i < n; i++) { m[i] = new Array(n).fill(0); m[i][i] = s }
    return m
  }

  _addMat(A, B) {
    const C = []
    for (let i = 0; i < A.length; i++) {
      C[i] = []
      for (let j = 0; j < A.length; j++) C[i][j] = A[i][j] + B[i][j]
    }
    return C
  }

  _FPFt(F, P) {
    // F·P·Fᵀ for 6×6
    const FP = this._eye(6, 0)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++)
        for (let k = 0; k < 6; k++)
          FP[i][j] += F[i][k] * P[k][j]

    const R = this._eye(6, 0)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++)
        for (let k = 0; k < 6; k++)
          R[i][j] += FP[i][k] * F[j][k]  // Fᵀ
    return R
  }
}


// ============================================================================
// IMAGE HELPERS — Grass / pitch verification
// ============================================================================

function rgbToHsv(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b)
  const d = max - min
  let h = 0
  if (d) {
    if (max === r) h = ((g - b) / d) % 6
    else if (max === g) h = (b - r) / d + 2
    else h = (r - g) / d + 4
    h /= 6
  }
  return [h < 0 ? h + 1 : h, max ? d / max : 0, max]
}

/**
 * Check if the area surrounding a bbox is green (pitch).
 * Ball in the lower half of the frame should be on or near the pitch.
 * If the surrounding area isn't green, it's likely a scoreboard/crowd detection.
 */
function grassScore(image, bbox) {
  const [x, y, w, h] = bbox
  const { width, height, data } = image
  const pad = FILTER_CONFIG.grass_sample_pad
  let greenPixels = 0, totalPixels = 0

  for (let dy = -pad; dy < h + pad; dy++) {
    for (let dx = -pad; dx < w + pad; dx++) {
      // Skip interior of bbox — only sample the border region
      if (dx >= 0 && dx < w && dy >= 0 && dy < h) continue
      const px = x + dx, py = y + dy
      if (px < 0 || py < 0 || px >= width || py >= height) continue
      const i = (py * width + px) * 4
      const [hue, sat, val] = rgbToHsv(data[i] / 255, data[i + 1] / 255, data[i + 2] / 255)
      if (hue > FILTER_CONFIG.grass_hue_min && hue < FILTER_CONFIG.grass_hue_max &&
          sat > FILTER_CONFIG.grass_sat_min && val > FILTER_CONFIG.grass_val_min) {
        greenPixels++
      }
      totalPixels++
    }
  }
  return totalPixels ? greenPixels / totalPixels : 0
}


// ============================================================================
// DETECTION FILTERS — Applied to raw RF-DETR output
// ============================================================================

/**
 * Size + aspect ratio filter.
 * Rejects shoes (too large/elongated) and noise (too small).
 */
function validGeometry(b) {
  const area = b[2] * b[3]
  const ar = b[2] / Math.max(b[3], 1)
  return area >= FILTER_CONFIG.min_bbox_area &&
         area <= FILTER_CONFIG.max_bbox_area &&
         ar >= FILTER_CONFIG.aspect_ratio_min &&
         ar <= FILTER_CONFIG.aspect_ratio_max
}

/**
 * Frame edge exclusion.
 * Rejects detections in broadcast bars, scoreboards at edges.
 */
function validPosition(b, image) {
  const cx = b[0] + b[2] / 2
  const cy = b[1] + b[3] / 2
  const { width, height } = image

  if (cy > height - FILTER_CONFIG.exclude_bottom_pixels) return false
  if (cy < height * FILTER_CONFIG.exclude_top_ratio) return false
  if (cx < width * FILTER_CONFIG.exclude_left_ratio) return false
  if (cx > width * (1 - FILTER_CONFIG.exclude_right_ratio)) return false

  return true
}

/**
 * Grass/pitch proximity check.
 * Detections in the lower portion of the frame must be near green pitch.
 * This filters out scoreboard overlays, logos, and crowd objects.
 */
function validGrass(b, image) {
  const cy = b[1] + b[3] / 2
  // Only enforce grass check for lower portion of frame
  if (cy <= image.height * FILTER_CONFIG.grass_check_zone_ratio) return true
  return grassScore(image, b) >= FILTER_CONFIG.grass_threshold
}

/**
 * Shoe likelihood scorer.
 * Detections near the feet of detected players are likely shoes.
 * Returns a score — higher means more likely to be a shoe.
 */
function computeShoeScore(ballBox, players, image) {
  if (!players || players.length === 0) return 0

  const [bx, by, bw, bh] = ballBox
  const bcx = bx + bw / 2
  const bcy = by + bh / 2
  let score = 0

  for (const p of players) {
    const [px, py, pw, ph] = p
    const pcx = px + pw / 2
    // Foot center: near the bottom of the player bbox
    const footY = py + ph * FILTER_CONFIG.shoe_foot_zone_ratio
    const dist = Math.hypot(bcx - pcx, bcy - footY)

    // Close to player feet → likely a shoe
    if (dist < pw * FILTER_CONFIG.shoe_proximity_ratio) {
      score += 1.0

      // Extra penalty if ball bbox is within the bottom portion of the player bbox
      if (bcy > py + ph * FILTER_CONFIG.shoe_bottom_zone_ratio) {
        score += 0.5
      }
    }

    // Slight penalty for being very close horizontally even if not exactly at feet
    const horizDist = Math.abs(bcx - pcx)
    if (horizDist < pw * 0.3 && bcy > py + ph * 0.7) {
      score += 0.3
    }
  }

  return score
}

/**
 * Two-tier confidence filter.
 * 
 * STRICT globally — rejects low-confidence false positives.
 * LENIENT inside the predicted search zone — allows re-acquisition of
 * aerial ball that returns with lower confidence.
 */
function passesConfidence(conf, cx, cy, searchZone) {
  if (searchZone) {
    const dist = Math.hypot(cx - searchZone.x, cy - searchZone.y)
    if (dist <= searchZone.radius) {
      return conf >= FILTER_CONFIG.confidence_search_zone
    }
  }
  return conf >= FILTER_CONFIG.confidence_global
}

/**
 * Maximum jump filter.
 * A ball can't teleport — if a detection appears far from the last known
 * position, it's almost certainly a false positive.
 */
function withinMaxJump(cx, cy, lastPos) {
  if (!lastPos) return true  // No history — allow anything
  return Math.hypot(cx - lastPos.x, cy - lastPos.y) <= FILTER_CONFIG.max_motion_px
}


// ============================================================================
// BALL TRACKER — State machine with Kalman filter
// ============================================================================
// 
// States:
//   NO_TRACK   → Waiting for first confident detection
//   TENTATIVE  → Got initial detection, need confirmation
//   TRACKING   → Active tracking (detection + Kalman update each frame)
//   PREDICTING → Detection lost, using Kalman prediction (grace period)
// 
// Transitions:
//   NO_TRACK  → TENTATIVE:  first detection passes all filters
//   TENTATIVE → TRACKING:   enough detections in confirmation window
//   TENTATIVE → NO_TRACK:   confirmation window expired
//   TRACKING  → TRACKING:   detection found near predicted position
//   TRACKING  → PREDICTING: no detection this frame
//   PREDICTING→ TRACKING:   detection re-acquired in search zone
//   PREDICTING→ NO_TRACK:   grace period exceeded

class BallTracker {
  constructor() {
    this.kalman = new KalmanFilter(
      FILTER_CONFIG.kalman_process_noise,
      FILTER_CONFIG.kalman_measurement_noise,
      FILTER_CONFIG.kalman_gravity,
    )

    this.state = 'NO_TRACK'
    this.framesLost = 0
    this.lastPosition = null      // { x, y } — last known (detected or predicted)
    this.lastDetectedPos = null   // { x, y } — last actually detected position
    this.lastBbox = null          // [x, y, w, h] of last detection
    this.lastConf = 0
    this.shoeFrames = 0           // Consecutive shoe-like detections

    // Tentative track confirmation
    this.candidateCount = 0
    this.candidateFrameStart = -1
    this.candidateLastDet = null
  }

  /**
   * Get the search zone for two-tier confidence.
   * Returns { x, y, radius } around predicted position, or null.
   */
  getSearchZone() {
    if (this.state === 'NO_TRACK' || !this.lastPosition) return null

    const speed = this.kalman.initialized ? this.kalman.getSpeed() : 0
    const radius = Math.max(
      FILTER_CONFIG.min_search_radius,
      Math.min(speed * FILTER_CONFIG.search_radius_multiplier, FILTER_CONFIG.max_search_radius)
    )

    return { x: this.lastPosition.x, y: this.lastPosition.y, radius }
  }

  /**
   * Main per-frame update.
   * 
   * @param {Array} dets  - Filtered detection bboxes [[x,y,w,h], ...]
   * @param {Array} confs - Corresponding confidences
   * @param {Array} players - Player bboxes for shoe detection
   * @param {Object} image - { width, height, data }
   * @param {number} frameIdx - Current frame index (for tentative window)
   * @returns {{ bboxes: Array, confidences: Array }}
   */
  update(dets, confs, players, image, frameIdx) {
    const debug = FILTER_CONFIG.debug

    switch (this.state) {

      // ════════════════════════════════════════
      case 'NO_TRACK': {
        if (dets.length === 0) return this._empty()

        const best = this._pickBest(dets, confs)

        if (FILTER_CONFIG.tentative_detections <= 1) {
          // Immediate track start
          return this._startTracking(best.bbox, best.conf)
        }

        // Enter tentative state
        this.state = 'TENTATIVE'
        this.candidateCount = 1
        this.candidateFrameStart = frameIdx
        this.candidateLastDet = best
        this.lastPosition = { x: best.cx, y: best.cy }
        if (debug) console.log(`[BALL] NO_TRACK → TENTATIVE (frame ${frameIdx})`)
        return this._empty()
      }

      // ════════════════════════════════════════
      case 'TENTATIVE': {
        // Check if confirmation window expired
        if (frameIdx - this.candidateFrameStart > FILTER_CONFIG.tentative_window) {
          this.state = 'NO_TRACK'
          this.candidateCount = 0
          if (debug) console.log(`[BALL] TENTATIVE expired → NO_TRACK`)
          // Still try this frame's detections as a fresh start
          if (dets.length > 0) {
            const best = this._pickBest(dets, confs)
            this.state = 'TENTATIVE'
            this.candidateCount = 1
            this.candidateFrameStart = frameIdx
            this.candidateLastDet = best
            this.lastPosition = { x: best.cx, y: best.cy }
          }
          return this._empty()
        }

        if (dets.length > 0) {
          const best = this._pickBest(dets, confs)
          this.candidateCount++
          this.candidateLastDet = best
          this.lastPosition = { x: best.cx, y: best.cy }

          if (this.candidateCount >= FILTER_CONFIG.tentative_detections) {
            if (debug) console.log(`[BALL] TENTATIVE → TRACKING (confirmed with ${this.candidateCount} detections)`)
            return this._startTracking(best.bbox, best.conf)
          }
        }

        return this._empty()
      }

      // ════════════════════════════════════════
      case 'TRACKING': {
        // Kalman predict (advance state by one frame)
        const [predX, predY] = this.kalman.predict()

        if (dets.length > 0) {
          // ── Shoe check ──
          const best = this._closestTo(dets, confs, predX, predY)
          const shoeScore = computeShoeScore(best.bbox, players, image)

          if (shoeScore >= FILTER_CONFIG.shoe_score_threshold) {
            this.shoeFrames++
            if (this.shoeFrames > FILTER_CONFIG.shoe_lock_frames) {
              if (debug) console.log(`[BALL] Shoe lock active (${this.shoeFrames} frames)`)
              // Don't use this detection, treat as if no detection
              this.state = 'PREDICTING'
              this.framesLost = 1
              this.lastPosition = { x: predX, y: predY }
              return this._predicted(predX, predY)
            }
          } else {
            this.shoeFrames = 0
          }

          // ── Kalman update with measurement ──
          const [corrX, corrY] = this.kalman.update(best.cx, best.cy)

          this.lastPosition = { x: corrX, y: corrY }
          this.lastDetectedPos = { x: corrX, y: corrY }
          this.lastBbox = best.bbox
          this.lastConf = best.conf
          this.framesLost = 0

          return {
            bboxes: [best.bbox],
            confidences: [best.conf],
          }
        }

        // ── No detection — enter PREDICTING ──
        this.state = 'PREDICTING'
        this.framesLost = 1
        this.lastPosition = { x: predX, y: predY }
        if (debug) console.log(`[BALL] TRACKING → PREDICTING (lost detection)`)

        return this._predicted(predX, predY)
      }

      // ════════════════════════════════════════
      case 'PREDICTING': {
        // Kalman predict (continues trajectory with velocity + gravity)
        const [predX, predY] = this.kalman.predict()
        this.framesLost++

        if (dets.length > 0) {
          // ── Try to re-acquire near predicted position ──
          const searchRadius = this._getSearchRadius()
          const nearby = []
          const nearbyConfs = []

          for (let i = 0; i < dets.length; i++) {
            const cx = dets[i][0] + dets[i][2] / 2
            const cy = dets[i][1] + dets[i][3] / 2
            if (Math.hypot(cx - predX, cy - predY) <= searchRadius) {
              nearby.push(dets[i])
              nearbyConfs.push(confs[i])
            }
          }

          if (nearby.length > 0) {
            const best = this._pickBest(nearby, nearbyConfs)
            const [corrX, corrY] = this.kalman.update(best.cx, best.cy)

            this.state = 'TRACKING'
            this.framesLost = 0
            this.shoeFrames = 0
            this.lastPosition = { x: corrX, y: corrY }
            this.lastDetectedPos = { x: corrX, y: corrY }
            this.lastBbox = best.bbox
            this.lastConf = best.conf

            if (debug) console.log(`[BALL] PREDICTING → TRACKING (re-acquired after ${this.framesLost} frames)`)

            return {
              bboxes: [best.bbox],
              confidences: [best.conf],
            }
          }
        }

        // ── Check grace period ──
        this.lastConf *= FILTER_CONFIG.confidence_decay

        if (this.framesLost > FILTER_CONFIG.max_missing_frames ||
            this.lastConf < FILTER_CONFIG.min_predicted_confidence) {
          if (debug) console.log(`[BALL] PREDICTING → NO_TRACK (lost for ${this.framesLost} frames)`)
          this._reset()
          return this._empty()
        }

        // ── Still predicting — output predicted position ──
        // Clamp to frame bounds
        const clampedX = Math.max(0, Math.min(predX, image.width))
        const clampedY = Math.max(0, Math.min(predY, image.height))
        this.lastPosition = { x: clampedX, y: clampedY }

        return this._predicted(clampedX, clampedY)
      }

      default:
        return this._empty()
    }
  }

  // ── Helpers ──

  _pickBest(bboxes, confs) {
    let best = 0
    for (let i = 1; i < confs.length; i++) {
      if (confs[i] > confs[best]) best = i
    }
    const b = bboxes[best]
    return {
      bbox: b,
      conf: confs[best],
      cx: b[0] + b[2] / 2,
      cy: b[1] + b[3] / 2,
    }
  }

  _closestTo(bboxes, confs, px, py) {
    let closest = 0
    let minDist = Infinity
    for (let i = 0; i < bboxes.length; i++) {
      const cx = bboxes[i][0] + bboxes[i][2] / 2
      const cy = bboxes[i][1] + bboxes[i][3] / 2
      const d = Math.hypot(cx - px, cy - py)
      if (d < minDist) { minDist = d; closest = i }
    }
    const b = bboxes[closest]
    return {
      bbox: b,
      conf: confs[closest],
      cx: b[0] + b[2] / 2,
      cy: b[1] + b[3] / 2,
    }
  }

  _getSearchRadius() {
    const speed = this.kalman.getSpeed()
    return Math.max(
      FILTER_CONFIG.min_search_radius,
      Math.min(speed * FILTER_CONFIG.search_radius_multiplier, FILTER_CONFIG.max_search_radius)
    )
  }

  _startTracking(bbox, conf) {
    const cx = bbox[0] + bbox[2] / 2
    const cy = bbox[1] + bbox[3] / 2
    this.kalman.initialize(cx, cy)
    this.state = 'TRACKING'
    this.framesLost = 0
    this.shoeFrames = 0
    this.lastPosition = { x: cx, y: cy }
    this.lastDetectedPos = { x: cx, y: cy }
    this.lastBbox = bbox
    this.lastConf = conf
    this.candidateCount = 0
    return { bboxes: [bbox], confidences: [conf] }
  }

  _predicted(px, py) {
    // Construct a synthetic bbox at predicted position using last known size
    const w = this.lastBbox ? this.lastBbox[2] : 20
    const h = this.lastBbox ? this.lastBbox[3] : 20
    const bbox = [px - w / 2, py - h / 2, w, h]
    return {
      bboxes: [bbox],
      confidences: [Math.max(this.lastConf, FILTER_CONFIG.min_predicted_confidence)],
    }
  }

  _empty() {
    return { bboxes: [], confidences: [] }
  }

  _reset() {
    this.state = 'NO_TRACK'
    this.framesLost = 0
    this.shoeFrames = 0
    this.lastPosition = null
    this.lastDetectedPos = null
    this.lastBbox = null
    this.lastConf = 0
    this.candidateCount = 0
    this.kalman = new KalmanFilter(
      FILTER_CONFIG.kalman_process_noise,
      FILTER_CONFIG.kalman_measurement_noise,
      FILTER_CONFIG.kalman_gravity,
    )
  }
}


// ============================================================================
// TRANSFORM STREAM — Drop-in replacement, same interface as original
// ============================================================================

/**
 * Creates a TransformStream that filters and tracks ball detections.
 * 
 * Input:  data with data.ball.bboxes, data.ball.confidences, data.players.bboxes, data.image
 * Output: data with data.ball replaced by filtered/tracked result
 * 
 * @returns {TransformStream}
 */
export function ballFilterStream() {
  let tracker = new BallTracker()
  let lastTime = 0
  let frameIdx = 0

  return new TransformStream({
    async transform(data, controller) {
      // ── Scene cut detection: reset tracker on large time jumps ──
      if (Math.abs(data.time - lastTime) > FILTER_CONFIG.scene_cut_time_gap) {
        tracker = new BallTracker()
        frameIdx = 0
        if (FILTER_CONFIG.debug) console.log(`[BALL] Scene cut detected (gap=${(data.time - lastTime).toFixed(2)}s) → reset tracker`)
      }
      lastTime = data.time

      const image = data.image
      const players = data.players?.bboxes || []
      const rawBboxes = data.ball?.bboxes || []
      const rawConfs = data.ball?.confidences || []

      // ── Get search zone from tracker (for two-tier confidence) ──
      const searchZone = tracker.getSearchZone()

      // ── Apply all filters to raw detections ──
      const filteredBboxes = []
      const filteredConfs = []

      for (let i = 0; i < rawBboxes.length; i++) {
        const b = rawBboxes[i]
        const conf = rawConfs[i]
        const cx = b[0] + b[2] / 2
        const cy = b[1] + b[3] / 2

        // Filter 1: Two-tier confidence
        if (!passesConfidence(conf, cx, cy, searchZone)) {
          if (FILTER_CONFIG.debug) console.log(`  [CONF REJECT] conf=${conf.toFixed(3)}`)
          continue
        }

        // Filter 2: Size + aspect ratio
        if (!validGeometry(b)) {
          if (FILTER_CONFIG.debug) console.log(`  [GEOM REJECT] area=${(b[2]*b[3]).toFixed(0)} ar=${(b[2]/b[3]).toFixed(2)}`)
          continue
        }

        // Filter 3: Frame edge exclusion
        if (!validPosition(b, image)) {
          if (FILTER_CONFIG.debug) console.log(`  [POS REJECT] cx=${cx.toFixed(0)} cy=${cy.toFixed(0)}`)
          continue
        }

        // Filter 4: Grass/pitch check (lower frame only)
        if (!validGrass(b, image)) {
          if (FILTER_CONFIG.debug) console.log(`  [GRASS REJECT] at cy=${cy.toFixed(0)}`)
          continue
        }

        // Filter 5: Max jump distance (temporal consistency)
        if (!withinMaxJump(cx, cy, tracker.lastDetectedPos)) {
          if (FILTER_CONFIG.debug) console.log(`  [JUMP REJECT] dist=${tracker.lastDetectedPos ? Math.hypot(cx - tracker.lastDetectedPos.x, cy - tracker.lastDetectedPos.y).toFixed(0) : 'N/A'}`)
          continue
        }

        filteredBboxes.push(b)
        filteredConfs.push(conf)
      }

      if (FILTER_CONFIG.debug && rawBboxes.length > 0) {
        console.log(`[BALL] Frame ${frameIdx}: raw=${rawBboxes.length} → filtered=${filteredBboxes.length} | state=${tracker.state}`)
      }

      // ── Pass filtered detections to tracker ──
      const tracked = tracker.update(filteredBboxes, filteredConfs, players, image, frameIdx)

      // ── Write result in same format as original ──
      data.ball = {
        bboxes: tracked.bboxes,
        confidences: tracked.confidences,
        classIds: tracked.bboxes.map(() => 37),
      }

      frameIdx++
      controller.enqueue(data)
    }
  })
}

export default ballFilterStream
