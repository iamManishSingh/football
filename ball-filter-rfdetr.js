/** @import { BBOX } from "../../utils/math.js" */

// ================= CONFIG =================

export const FILTER_CONFIG = {
  min_bbox_area: 14,
  max_bbox_area: 6000,
  aspect_ratio_min: 0.6,
  aspect_ratio_max: 1.7,
  exclude_bottom_pixels: 20,

  grass_threshold: 0.12,

  max_missing_frames: 8,
  max_motion_px: 260,

  shoe_lock_frames: 3
}

// ================= IMAGE HELPERS =================

function rgbToHsv(r,g,b){
  const max=Math.max(r,g,b), min=Math.min(r,g,b)
  const d=max-min
  let h=0
  if(d){
    if(max===r) h=((g-b)/d)%6
    else if(max===g) h=(b-r)/d+2
    else h=(r-g)/d+4
    h/=6
  }
  return [h<0?h+1:h, max?d/max:0, max]
}

function grassScore(image,bbox){
  const [x,y,w,h]=bbox
  const {width,height,data}=image
  let g=0,t=0
  for(let dy=-6;dy<h+6;dy++){
    for(let dx=-6;dx<w+6;dx++){
      if(dx>=0&&dx<w&&dy>=0&&dy<h) continue
      const px=x+dx, py=y+dy
      if(px<0||py<0||px>=width||py>=height) continue
      const i=(py*width+px)*4
      const [hue,sat,val]=rgbToHsv(data[i]/255,data[i+1]/255,data[i+2]/255)
      if(hue>0.25&&hue<0.45&&sat>0.2&&val>0.2) g++
      t++
    }
  }
  return t?g/t:0
}

// ================= SHOE FILTER =================

function shoeLikelihood(ballBox, players, image){
  if(!players) return 0
  const [bx,by,bw,bh] = ballBox
  const bcx = bx + bw/2
  const bcy = by + bh/2

  let score = 0

  for(const p of players){
    const [px,py,pw,ph] = p
    const pcx = px + pw/2
    const pcy = py + ph*0.9

    const dist = Math.hypot(bcx-pcx, bcy-pcy)

    if(dist < pw*0.4) score += 1
    if(bcy > image.height*0.75) score += 0.5
  }
  return score
}

// ================= BASIC FILTERS =================

function validGeometry(b){
  const area=b[2]*b[3]
  const ar=b[2]/b[3]
  return area>FILTER_CONFIG.min_bbox_area &&
         area<FILTER_CONFIG.max_bbox_area &&
         ar>FILTER_CONFIG.aspect_ratio_min &&
         ar<FILTER_CONFIG.aspect_ratio_max
}

function validPosition(b,image){
  return b[1]+b[3]/2 < image.height - FILTER_CONFIG.exclude_bottom_pixels
}

// ================= TRACKER =================

class BallTracker {
  constructor(){ this.track=null; this.shoeFrames=0 }

  center(b){ return [b[0]+b[2]/2,b[1]+b[3]/2] }

  update(dets,confs,players,image){

    // ---- DETECTION WINS ----
    if(dets.length>0){
      let best=0
      for(let i=1;i<confs.length;i++)
        if(confs[i]>confs[best]) best=i

      const b=dets[best]
      const c=this.center(b)

      // ---- SHOE TEMPORAL FILTER ----
      const shoeScore = shoeLikelihood(b, players, image)
      if(shoeScore > 1){
        this.shoeFrames++
      } else {
        this.shoeFrames = 0
      }

      if(this.shoeFrames > FILTER_CONFIG.shoe_lock_frames){
        return {bboxes:[],confidences:[]}
      }

      if(this.track){
        const dx=c[0]-this.track.cx
        const dy=c[1]-this.track.cy
        if(Math.hypot(dx,dy) < FILTER_CONFIG.max_motion_px){
          this.track.vx = dx
          this.track.vy = dy
        }
      } else {
        this.track={vx:0,vy:0}
      }

      this.track={
        ...this.track,
        bbox:b,
        cx:c[0],
        cy:c[1],
        conf:confs[best],
        miss:0
      }

      return {bboxes:[b],confidences:[confs[best]]}
    }

    // ---- SHORT GAP FILL ----
    if(this.track){
      this.track.miss++
      this.track.conf*=0.88

      if(this.track.miss<=FILTER_CONFIG.max_missing_frames && this.track.conf>0.05){
        this.track.cx+=this.track.vx||0
        this.track.cy+=this.track.vy||0
        this.track.bbox=[
          this.track.cx-this.track.bbox[2]/2,
          this.track.cy-this.track.bbox[3]/2,
          this.track.bbox[2],
          this.track.bbox[3]
        ]
        return {bboxes:[this.track.bbox],confidences:[this.track.conf]}
      }

      this.track=null
    }

    return {bboxes:[],confidences:[]}
  }
}

// ================= STREAM =================

export function ballFilterStream(){
  let tracker=new BallTracker()
  let lastTime=0

  return new TransformStream({
    async transform(data,controller){

      if(Math.abs(data.time-lastTime)>1) tracker=new BallTracker()
      lastTime=data.time

      const image=data.image
      const players=data.players?.bboxes || []
      const raw=data.ball?.bboxes||[]
      const conf=data.ball?.confidences||[]

      const dets=[],confs=[]
      for(let i=0;i<raw.length;i++){
        if(validGeometry(raw[i]) && validPosition(raw[i],image)){
          const centerY=raw[i][1]+raw[i][3]/2

          if(centerY>image.height*0.6 &&
             grassScore(image,raw[i]) < FILTER_CONFIG.grass_threshold)
            continue

          dets.push(raw[i])
          confs.push(conf[i])
        }
      }

      const tracked=tracker.update(dets,confs,players,image)

      data.ball={
        bboxes:tracked.bboxes,
        confidences:tracked.confidences,
        classIds:tracked.bboxes.map(()=>37)
      }

      controller.enqueue(data)
    }
  })
}

export default ballFilterStream
