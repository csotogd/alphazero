import { env } from 'cloudflare:workers';
import { newGame, play, pass, type Game, type Move } from '@/lib/game/engine';

type Room = {code:string; state:string; host_hash:string; guest_hash:string|null; host_name:string; guest_name:string|null; revision:number; rematch:number; created_at:number};
const reply=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
const fail=(message:string,status=400)=>reply({error:message},status);
const hash=async(token:string)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token)))).map(b=>b.toString(16).padStart(2,'0')).join('');
function view(room:Room,player:number) {return {code:room.code,game:JSON.parse(room.state),names:[room.host_name,room.guest_name || 'Waiting for a friend'],ready:!!room.guest_hash,revision:room.revision,rematch:room.rematch,player};}
const db=()=>{if(!env.DB)throw new Error('Online rooms are temporarily unavailable. You can still play locally.');return env.DB;};
async function roomFor(code:string) {return db().prepare('SELECT * FROM rooms WHERE code = ? AND created_at > ?').bind(code,Date.now()-7*86400000).first<Room>();}
async function identify(room:Room,token:string) {const h=await hash(token);return h===room.host_hash?0:h===room.guest_hash?1:-1;}
export async function GET(request:Request) {
  try {
    const code=new URL(request.url).searchParams.get('code')||'';
    const room=await roomFor(code);if(!room)return fail('Room not found or expired. Rooms last seven days.',404);
    const player=await identify(room,request.headers.get('Authorization')?.replace(/^Bearer /,'')||'');
    if(player<0)return fail('Join this room to view the game.',403);
    return reply(view(room,player));
  } catch {return fail('Could not load the room. Please try again.',503);}
}
export async function POST(request:Request) {
  try {
    if(request.headers.get('Origin') && request.headers.get('Origin')!==new URL(request.url).origin)return fail('Invalid origin.',403);
    if(Number(request.headers.get('Content-Length')||0)>4096)return fail('Request too large.',413);
    const raw=await request.text();if(raw.length>4096)return fail('Request too large.',413);
    const body=JSON.parse(raw);
    if(!body || typeof body!=='object')return fail('Invalid request.');
    const {action}=body;
    const name=typeof body.name==='string'?body.name.trim().slice(0,24):'';
    if(action==='create') {
      if(!name)return fail('Enter your name first.');
      const token=crypto.randomUUID();const hostHash=await hash(token);
      const alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
      for(let attempt=0;attempt<4;attempt++) {
        const code=Array.from(crypto.getRandomValues(new Uint8Array(6)),v=>alphabet[v%alphabet.length]).join('');
        const result=await db().prepare('INSERT OR IGNORE INTO rooms (code,state,host_hash,host_name,revision,rematch,created_at) VALUES (?,?,?,?,0,0,?)').bind(code,JSON.stringify(newGame()),hostHash,name,Date.now()).run();
        if(result.meta.changes){const room=await roomFor(code);return reply({...view(room!,0),token});}
      }
      return fail('Could not create a room. Please try again.',503);
    }
    const code=typeof body.code==='string'?body.code.toUpperCase():'';
    if(!/^[A-Z2-9]{6}$/.test(code))return fail('Enter a six-character room code.');
    const room=await roomFor(code);if(!room)return fail('Room not found or expired.',404);
    if(action==='join') {
      if(!name)return fail('Enter your name first.');
      if(room.guest_hash)return fail('This room already has two players.',409);
      const token=crypto.randomUUID();
      const result=await db().prepare('UPDATE rooms SET guest_hash=?,guest_name=?,revision=revision+1 WHERE code=? AND guest_hash IS NULL').bind(await hash(token),name,code).run();
      if(!result.meta.changes)return fail('Another player just joined this room.',409);
      return reply({...view((await roomFor(code))!,1),token});
    }
    const token=request.headers.get('Authorization')?.replace(/^Bearer /,'')||'';
    const player=await identify(room,token);if(player<0)return fail('Your room session is invalid. Rejoin or create a new room.',403);
    if(!room.guest_hash)return fail('Wait for your friend to join.');
    if(body.revision!==room.revision)return fail('The board changed. Please try your move again.',409);
    let game=JSON.parse(room.state) as Game;let rematch=room.rematch;
    if(action==='rematch') {
      if(!game.finished)return fail('Finish this game before requesting a rematch.');
      rematch|=(1<<player);if(rematch===3){game=newGame();rematch=0;}
    } else {
      if(game.turn!==player)return fail('It is your opponent’s turn.',409);
      if(action==='move') {
        if(!body.move || typeof body.move.piece!=='string')return fail('Choose a valid piece.');
        game=play(game,body.move as Move);
      } else if(action==='pass')game=pass(game);
      else return fail('Unknown action.');
    }
    const result=await db().prepare('UPDATE rooms SET state=?,rematch=?,revision=revision+1 WHERE code=? AND revision=?').bind(JSON.stringify(game),rematch,code,room.revision).run();
    if(!result.meta.changes)return fail('The board changed. Please try again.',409);
    return reply(view((await roomFor(code))!,player));
  } catch(error) {return fail(error instanceof SyntaxError?'Invalid request.':error instanceof Error?error.message:'Something went wrong.',400);}
}
