import assert from 'node:assert/strict';
import {newGame,validate,shape,PIECES,SIZE,pass} from '../lib/game/engine.ts';
let input='';for await(const chunk of process.stdin)input+=chunk;
const cases=JSON.parse(input);
for(const item of cases){
 const game={...newGame(),...item.state};const expected=new Set();
 for(const piece of game.remaining[game.turn]){
  const seen=new Set();for(const flipped of [false,true])for(let rotation=0;rotation<4;rotation++){
   const key=JSON.stringify(shape(piece,rotation,flipped).map(c=>c.join()).sort());if(seen.has(key))continue;seen.add(key);
   for(let y=0;y<SIZE;y++)for(let x=0;x<SIZE;x++){const m={piece,rotation,flipped,x,y};if(!validate(game,m))expected.add(JSON.stringify(m));}
  }
 }
 const actual=new Set(item.legal.filter(m=>!m.pass).map(m=>JSON.stringify(m)));
 assert.deepEqual(actual,expected);
 if(item.legal.some(m=>m.pass)){assert.equal(expected.size,0);assert.doesNotThrow(()=>pass(game));}
}
console.log(`${cases.length} positions: complete legal-action parity verified.`);
