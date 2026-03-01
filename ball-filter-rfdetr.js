/** @import { BBOX } from "../../utils/math.js" */

/**
 * ⚽ Ball Detection Post-Processing for RF-DETR — Lightweight Edition
 *
 * Drop-in replacement — same interface as original:
 *   ballFilterStream() → TransformStream
 *   data.ball = { bboxes, confidences, classIds }
 *
 * Goal: reject obvious false positives (shoes, static white objects like corner
 * flags / pitch logos / ads) without any complex tracking that over-filters
 * real detections. The pre-trained model is already good; we just need to
 * remove the noise.
 *
 * Three lightweight filters applied each frame:
 *  1. Confidence threshold  — drop weak detections
 *  2. Size + aspect ratio   — ball is small & ~circular; shoes are elongated
 *  3. Static-object guard   — objects stuck in the same spot for N frames
 *                             (corner flags, logos, pitch markings, shoe soles)
 *
 * Output per frame: highest-confidence detection that survives, or empty.
 */

// ============================================================================
// CONFIG
// ============================================================================

export const FILTER_CONFIG = {
  // ── Confidence ──────────────────────────────────────────────────────────────
  // Lower = more detections (catches blurry/occluded ball).
  // Raise if you see too many false positives.
  confidence_threshold: 0.25,

  // ── Size (frame-area ratios — resolution-independent) ───────────────────────
  // At 1920×1080 the frame is ~2 073 600 px².
  //   min: 0.000015 → ~31 px²   (rejects single-pixel noise)
  //   max: 0.006    → ~12 400 px²  (rejects large shoe / player regions)
  min_ball_area_ratio: 0.000015,
  max_ball_area_ratio: 0.0048,

  // ── Aspect ratio (width / height) — ball is roughly circular ────────────────
  // Shoes and pitch markings are much wider than tall.
  min_aspect_ratio: 0.50,
  max_aspect_ratio: 1.9,

  // ── Static-object rejection ─────────────────────────────────────────────────
  // Tracks recent detection positions in a rolling window.
  // If a candidate appears in >= static_max_hit_ratio of recent frames
  // at nearly the same spot it is considered static and rejected.
  enable_static_rejection: true,
  static_history_frames: 14,       // rolling window length (frames)
  static_cluster_radius_px: 28,    // detections within this px radius = same spot
  static_max_hit_ratio: 0.72,      // fraction of window → reject
  static_warmup_frames: 4,         // don't reject until we have this much history

  // ── Scene cut ───────────────────────────────────────────────────────────────
  scene_cut_time_gap: 1.5,         // seconds; resets static history on clip cuts

  // ── Debug ───────────────────────────────────────────────────────────────────
  debug: false,
}


// ============================================================================
// PER-DETECTION FILTERS
// ============================================================================

/**
 * Size + aspect-ratio filter.
 * Uses frame-area ratios so thresholds work at any resolution.
 */
function validGeometry(b, frameArea) {
  const area = b[2] * b[3]
  const ar   = b[2] / Math.max(b[3], 1)
  return (
    area >= FILTER_CONFIG.min_ball_area_ratio * frameArea &&
    area <= FILTER_CONFIG.max_ball_area_ratio * frameArea &&
    ar   >= FILTER_CONFIG.min_aspect_ratio &&
    ar   <= FILTER_CONFIG.max_aspect_ratio
  )
}

/** Simple confidence gate. */
function passesConfidence(conf) {
  return conf >= FILTER_CONFIG.confidence_threshold
}


// ============================================================================
// STATIC-OBJECT REJECTER
// ============================================================================
//
// Maintains a rolling list of recent detection centres.
// A candidate is "static" if the majority of recent frames had a detection
// within cluster_radius_px of the same spot — a strong sign it is a
// non-moving object (corner flag, pitch logo, ad board, shoe sole on grass).

class StaticRejecter {
  constructor() {
    /** @type {{ cx: number, cy: number, frameIdx: number }[]} */
    this._history = []
  }

  reset() {
    this._history = []
  }

  /**
   * Returns true if (cx, cy) looks like a static object.
   * Records the detection into the internal history regardless of the result.
   */
  isStatic(cx, cy, frameIdx) {
    const {
      static_history_frames: window,
      static_cluster_radius_px: radius,
      static_max_hit_ratio: maxRatio,
      static_warmup_frames: warmup,
    } = FILTER_CONFIG

    // Prune entries outside the rolling window
    const minFrame = frameIdx - window
    this._history = this._history.filter(h => h.frameIdx >= minFrame)

    // Count distinct frames that had a detection near this spot
    // (count BEFORE recording current frame so we don't inflate the ratio)
    const nearbyFrames = new Set()
    for (const h of this._history) {
      if (Math.hypot(h.cx - cx, h.cy - cy) <= radius) {
        nearbyFrames.add(h.frameIdx)
      }
    }

    // Record current detection
    this._history.push({ cx, cy, frameIdx })

    // Not enough history yet — don't reject
    const framesInWindow = frameIdx - minFrame
    if (framesInWindow < warmup) return false

    const hitRatio = nearbyFrames.size / framesInWindow
    if (FILTER_CONFIG.debug && hitRatio >= maxRatio) {
      console.log(
        `[BALL] static reject cx=${Math.round(cx)} cy=${Math.round(cy)} ` +
        `hitRatio=${hitRatio.toFixed(2)} (${nearbyFrames.size}/${framesInWindow} frames)`,
      )
    }
    return hitRatio >= maxRatio
  }
}


// ============================================================================
// TRANSFORM STREAM — drop-in replacement, same interface as original
// ============================================================================

/**
 * Creates a TransformStream that applies lightweight false-positive rejection.
 *
 * Input:  data.ball.{ bboxes, confidences }, data.image
 * Output: data.ball replaced with filtered { bboxes, confidences, classIds }
 *         At most one detection per frame (highest-confidence survivor).
 *
 * @returns {TransformStream}
 */
export function ballFilterStream() {
  const staticRejecter = new StaticRejecter()
  let lastTime = 0
  let frameIdx = 0

  return new TransformStream({
    transform(data, controller) {

      // Scene-cut: reset static history on large time gap (clip cuts, seeks)
      if (data.time != null && Math.abs(data.time - lastTime) > FILTER_CONFIG.scene_cut_time_gap) {
        staticRejecter.reset()
        frameIdx = 0
        if (FILTER_CONFIG.debug) {
          console.log(`[BALL] Scene cut (gap=${(data.time - lastTime).toFixed(2)}s) → static history reset`)
        }
      }
      if (data.time != null) lastTime = data.time

      const image     = data.image
      const frameArea = image.width * image.height

      const rawBboxes = data.ball?.bboxes      || []
      const rawConfs  = data.ball?.confidences || []

      // ── Apply per-detection filters ────────────────────────────────────────
      /** @type {{ bbox: number[], conf: number }[]} */
      const survivors = []

      for (let i = 0; i < rawBboxes.length; i++) {
        const b    = rawBboxes[i]
        const conf = rawConfs[i]
        const cx   = b[0] + b[2] / 2
        const cy   = b[1] + b[3] / 2

        // 1. Confidence threshold
        if (!passesConfidence(conf)) {
          if (FILTER_CONFIG.debug) console.log(`  [CONF]  reject conf=${conf.toFixed(3)}`)
          continue
        }

        // 2. Size + aspect ratio
        if (!validGeometry(b, frameArea)) {
          if (FILTER_CONFIG.debug) {
            const ar = (b[2] / Math.max(b[3], 1)).toFixed(2)
            console.log(`  [GEOM]  reject area=${(b[2] * b[3]).toFixed(0)} ar=${ar}`)
          }
          continue
        }

        // 3. Static-object guard
        if (FILTER_CONFIG.enable_static_rejection && staticRejecter.isStatic(cx, cy, frameIdx)) {
          if (FILTER_CONFIG.debug) console.log(`  [STATIC] reject cx=${Math.round(cx)} cy=${Math.round(cy)}`)
          continue
        }

        survivors.push({ bbox: b, conf })
      }

      if (FILTER_CONFIG.debug && rawBboxes.length > 0) {
        console.log(`[BALL] frame=${frameIdx} raw=${rawBboxes.length} survived=${survivors.length}`)
      }

      // ── Pick the single highest-confidence survivor ────────────────────────
      survivors.sort((a, b) => b.conf - a.conf)
      const winner = survivors[0] ?? null

      data.ball = {
        bboxes:      winner ? [winner.bbox] : [],
        confidences: winner ? [winner.conf] : [],
        classIds:    winner ? [37] : [],
      }

      frameIdx++
      controller.enqueue(data)
    },
  })
}

export default ballFilterStream
