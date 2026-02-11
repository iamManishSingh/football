import { combine } from "../utils/combine.js"
import { map, mapThrough, pipeThrough } from "../utils/map.js"
import { rfdtrInference, yoloInference } from "../utils/ring.js"
import { ringStream, burst } from "../utils/ring.js"

/** @import { ClipTimes, Pipeline } from '../pipeline.js' */
/** @import { DetectionData } from '../utils/onnx.js' */
/** @import { Frame } from '../utils/ffmpeg.js' */

import classifier from "../inference/classifier.js"
import createProcessingStream from "../processors/stream.js"
import { preProcessing } from "../utils/image/chroma.js"
// 🚀 ENHANCED: Import advanced ball filter for RF-DETR
import { ballFilterStream } from '../inference/ball/ball-filter-rfdetr.js'

// ============================================================================
// 🎛️ BALL FILTER TOGGLE
// ============================================================================
// Set to false to disable ball filtering (use raw RF-DETR detections)
// Set to true to enable advanced filtering (spatial, geometric, tracking, etc.)
export const ENABLE_BALL_FILTERS = true

/**
 * @param {{ batch?: number, clipTimes?: ClipTimes }} [options] 
 */
export const defaultPipeline = ({ batch = 16, clipTimes = [[0, Infinity]] } = {}) => ({
  /**
   * @returns {Pipeline['inference']}
   */
  get inference() {
    return inferencePipeline({ batch })
  },
  /**
   * @returns {Pipeline['processing']}
   */
  get processing() {
    return createProcessingStream(clipTimes)
  }
})

/**
 * @template T
 * @param {{batch?: number}} [options]
 * @returns {Pipeline['inference']}
 */
export function inferencePipeline({ batch = 16 } = {}) {
  const stage0 = ringStream(
    1080 * 1920 * 4 * 2 * 128,
    (item) => item.image.data,
    (frame, data) => (frame.image.data = data, frame)
  )
  /**
   * @typedef {Frame & {
   *   players: DetectionData;
   *   pitch: DetectionData & {keypoints: NonNullable<DetectionData['keypoints']>;};
   *   ball: DetectionData;
   * }} Stage1Data
   */

  /**
   * @type {TransformStream<NoInfer<Frame>, Stage1Data>}
   */
  const stage1 = combine({
    players: rfdtrInference('base', {
      classes: [ 1, 37 ],
      threshold: 0.1,
      batch: batch
    }),
    pitch: mapThrough(
      yoloInference('pitch-experimental', { batch: batch }), 
      filterKeypoints
    ),
    // 🚀 ENHANCED: Ball detection with RF-DETR + advanced filters
    ball: rfdtrInference('base', {
      classes: [ 37 ],           // Ball class
      threshold: 0.19,           // Lower threshold (filters handle false positives)
      batch: batch
    }),
  })
  
  // 🚀 ENHANCED: Conditionally apply ball filters based on toggle
  const stage1Filtered = ENABLE_BALL_FILTERS 
    ? pipeThrough(stage1, ballFilterStream())  // With filters
    : stage1                                    // Without filters (raw RF-DETR)
  
  /**
   * @typedef {Stage1Data & {
   *  ball: DetectionData,
   *  players: DetectionData,
   *  jerseys: DetectionData
   * }} Stage2Data
   */
  
  /**
   * @type {TransformStream<Stage1Data, Stage2Data>}
  **/
  const stage2 =
    // pipeThrough(
    //   new TransformStream({
    //     transform(inference, ctrl) {
          
    //       inference.image = preProcessing(inference.image)
    //       ctrl.enqueue(inference)
    //     }
    //   }),
      classifier({
        classes: {
          ball: [37],
          player: [1]
        },
        remap: { 1: 2, 37: 0 }
      })
    //)


  return pipeThrough(stage1Filtered, stage2)
}

/**
 * 
 * @param {DetectionData} pitch 
 * @returns 
 */
function filterKeypoints(pitch) {
  return pitch
  const keypoints = pitch.keypoints?.[0] 
    ?? Array.from({ length: 32 }, () => [ 0,0 ])

  const indexes = keypoints.keys()
    .toArray()
    .sort((a, b) => {
      const [,, confidenceA = 0 ] = keypoints[a]
      const [,, confidenceB = 0 ] = keypoints[b]

      return confidenceA - confidenceB
    })
  let remaining = keypoints.length

  for(const index of indexes) {
    const [x, y, confidence = 0 ] = keypoints[index]

    if(x === 0 || y === 0 || x === 1920 || y === 1080) {
      keypoints[index] = [ 0, 0 ]
      continue
    }

    if(confidence < 0.4 && remaining > 4) {
      keypoints[index] = [ 0, 0 ]
      remaining -= 1
    }
  }

  return Object.assign(pitch, {
    keypoints: [ keypoints ]
  })
}


export default defaultPipeline