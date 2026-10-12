import test from 'node:test';
import assert from 'node:assert/strict';
import {PIECES,SIZE,newGame,shape,validate,play,pass,findMove,score} from '../lib/game/engine.ts';
test('all twelve pieces have five distinct connected cells, across every orientation',()=>{
 assert.equal(Object.keys(PIECES).length,12);
 for(const p of Object.keys(PIECES))for(const f of [false,true])for(let r=0;r<4;r++){
  const cells=shape(p,r,f);assert.equal(cells.length,5);assert.equal(new Set(cells.map(c=>c.join())).size,5);
  const reached=[cells[0]];for(let i=0;i<reached.length;i++)for(const c of cells)if(!reached.includes(c) && Math.abs(c[0]-reached[i][0])+Math.abs(c[1]-reached[i][1])===1)reached.push(c);
  assert.equal(reached.length,5);
 }
});
test('first moves cover opposite corners; invalid coordinates and overlap are rejected',()=>{
 let g=newGame();assert.match(validate(g,{piece:'P',x:3,y:3,rotation:0,flipped:false}),/corner/);
 assert.match(validate(g,{piece:'P',x:NaN,y:0,rotation:0,flipped:false}),/Invalid/);
 assert.match(validate(g,{piece:'I',x:0,y:10,rotation:0,flipped:false}),/board/);
 g=play(g,{piece:'P',x:0,y:0,rotation:0,flipped:false});assert.equal(score(g,0),5);assert.equal(g.turn,1);
 assert.match(validate(g,{piece:'P',x:0,y:0,rotation:0,flipped:false}),/taken/);
 const m=findMove(g);assert.ok(m);g=play(g,m);assert.equal(score(g,1),5);assert.equal(g.board[SIZE*SIZE-1],1);
});
test('own edges prohibited; diagonal required; state updates immutable',()=>{
 const original=newGame();const first=play(original,{piece:'P',x:0,y:0,rotation:0,flipped:false});
 const g=play(first,findMove(first));assert.equal(original.board.filter(x=>x!==null).length,0);
 assert.match(validate(g,{piece:'I',x:2,y:0,rotation:0,flipped:false}),/edge/);
 assert.match(validate(g,{piece:'I',x:5,y:0,rotation:0,flipped:false}),/corner/);
 assert.equal(validate(g,{piece:'I',x:2,y:2,rotation:0,flipped:false}),null);
 assert.throws(()=>pass(g),/legal move/);
});
test('a full legal self-play game terminates with coherent scoring and no reused pieces',()=>{
 let g=newGame();let turns=0;while(!g.finished && turns++<30){const m=findMove(g);g=m?play(g,m):pass(g);}
 assert.ok(g.finished);assert.ok(turns<=26);assert.equal(g.board.filter(c=>c!==null).length,score(g,0)+score(g,1));
 assert.equal(g.history.length,g.moves);for(const p of [0,1])assert.equal(new Set(g.history.filter(m=>m.player===p).map(m=>m.piece)).size,12-g.remaining[p].length);
 assert.throws(()=>pass(g),/complete/);
});
