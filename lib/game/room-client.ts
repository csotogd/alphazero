import type {Game,Player} from './engine';
export type RoomView={code:string;game:Game;names:[string,string];ready:boolean;revision:number;player:Player;rematch:number;token?:string};
export type Session={code:string;token:string};
export async function roomRequest(body:Record<string,unknown>,session?:Session):Promise<RoomView> {
  const response=await fetch('/api/rooms',{method:'POST',headers:{'Content-Type':'application/json',...(session?{Authorization:`Bearer ${session.token}`}:{})},body:JSON.stringify(body)});
  const result=await response.json() as RoomView & {error?:string};if(!response.ok)throw new Error(result.error||'Could not connect. Try again.');return result;
}
export async function readRoom(session:Session,signal?:AbortSignal):Promise<RoomView> {
  const response=await fetch(`/api/rooms?code=${session.code}`,{headers:{Authorization:`Bearer ${session.token}`},cache:'no-store',signal});
  const result=await response.json() as RoomView & {error?:string};if(!response.ok)throw new Error(result.error||'Could not connect to the room.');return result;
}
