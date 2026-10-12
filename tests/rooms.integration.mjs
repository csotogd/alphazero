import assert from 'node:assert/strict';
import {findMove} from '../lib/game/engine.ts';
const base=process.env.TEST_BASE_URL||'http://127.0.0.1:5173';
async function post(body,token){const r=await fetch(`${base}/api/rooms`,{method:'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:`Bearer ${token}`}:{})},body:JSON.stringify(body)});return {status:r.status,data:await r.json()};}
const host=(await post({action:'create',name:'Test Violet'})).data;assert.ok(host.token);
const early=await post({action:'move',code:host.code,revision:host.revision,move:findMove(host.game)},host.token);assert.equal(early.status,400);
const guest=(await post({action:'join',code:host.code,name:'Test Lime'})).data;assert.ok(guest.token);assert.equal(guest.player,1);
assert.equal((await post({action:'join',code:host.code,name:'Third'})).status,409);
assert.equal((await fetch(`${base}/api/rooms?code=${host.code}`)).status,403);
assert.equal((await post({action:'move',code:host.code,revision:guest.revision,move:findMove(guest.game)},guest.token)).status,409);
assert.equal((await post({action:'move',code:host.code,revision:guest.revision,move:{piece:'P',x:7,y:7,rotation:0,flipped:false}},host.token)).status,400);
const body={action:'move',code:host.code,revision:guest.revision,move:findMove(guest.game)};
const simultaneous=await Promise.all([post(body,host.token),post(body,host.token)]);assert.deepEqual(simultaneous.map(r=>r.status).sort(),[200,409]);
let current=simultaneous.find(r=>r.status===200).data;assert.equal(current.game.moves,1);
let turns=0;while(!current.game.finished && turns++<30){const m=findMove(current.game);const r=await post({action:m?'move':'pass',code:host.code,revision:current.revision,move:m},current.game.turn===0?host.token:guest.token);assert.equal(r.status,200);current=r.data;}
assert.ok(current.game.finished);
let r=await post({action:'rematch',code:host.code,revision:current.revision},host.token);assert.equal(r.data.rematch,1);assert.ok(r.data.game.finished);
r=await post({action:'rematch',code:host.code,revision:r.data.revision},guest.token);assert.equal(r.data.game.moves,0);assert.equal(r.data.rematch,0);assert.equal(r.data.game.finished,false);
console.log('PASS: rooms, authentication, joining, turn checks, invalid moves, concurrent updates, complete game and mutual rematch.');
