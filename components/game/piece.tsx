import {shape} from '@/lib/game/engine';
export function Piece({piece,rotation=0,flipped=false}:{piece:string;rotation?:number;flipped?:boolean}) {
  const cells=shape(piece,rotation,flipped);
  const w=Math.max(...cells.map(c=>c[0]))+1,h=Math.max(...cells.map(c=>c[1]))+1;
  return <svg className="piece-svg" viewBox={`0 0 ${w*15} ${h*15}`} aria-hidden="true">{cells.map(([x,y])=><rect key={`${x},${y}`} x={x*15+.7} y={y*15+.7} width="13.6" height="13.6" rx="2.1" fill="currentColor"/>)}</svg>;
}
