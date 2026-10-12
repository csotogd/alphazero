/** Pure, deterministic rules. Shared by the browser, authoritative server and tests. */
export type Cell = readonly [number, number];
export type Player = 0 | 1;
export const SIZE = 12;
export const PIECES: Record<string, Cell[]> = {
  F: [[1,0],[2,0],[0,1],[1,1],[1,2]], I: [[0,0],[0,1],[0,2],[0,3],[0,4]],
  L: [[0,0],[0,1],[0,2],[0,3],[1,3]], P: [[0,0],[1,0],[0,1],[1,1],[0,2]],
  N: [[0,0],[0,1],[1,1],[1,2],[1,3]], T: [[0,0],[1,0],[2,0],[1,1],[1,2]],
  U: [[0,0],[2,0],[0,1],[1,1],[2,1]], V: [[0,0],[0,1],[0,2],[1,2],[2,2]],
  W: [[0,0],[0,1],[1,1],[1,2],[2,2]], X: [[1,0],[0,1],[1,1],[2,1],[1,2]],
  Y: [[0,0],[0,1],[1,1],[0,2],[0,3]], Z: [[0,0],[1,0],[1,1],[1,2],[2,2]],
};
export type Move = {piece: string; rotation: number; flipped: boolean; x: number; y: number};
export type Game = {board: (Player | null)[]; remaining: [string[],string[]]; turn: Player; passes: number; finished: boolean; moves: number; lastMove: number[]; history: {player: Player; piece: string; squares: number}[]};
export function newGame(): Game {
  return {board: Array(SIZE*SIZE).fill(null), remaining: [Object.keys(PIECES),Object.keys(PIECES)], turn: 0, passes: 0, finished: false, moves: 0, lastMove: [], history: []};
}
export function shape(piece: string, rotation = 0, flipped = false): Cell[] {
  let cells = (PIECES[piece] || []).map(([x,y]) => [flipped ? -x : x,y]);
  for (let i=0;i<((rotation%4)+4)%4;i++) cells = cells.map(([x,y])=>[-y,x]);
  const minX = Math.min(...cells.map(c=>c[0])), minY = Math.min(...cells.map(c=>c[1]));
  return cells.map(([x,y])=>[x-minX,y-minY] as Cell);
}
export function validate(game: Game, move: Move): string | null {
  if (game.finished) return 'This game is complete. Start a rematch!';
  if (!PIECES[move.piece] || !game.remaining[game.turn].includes(move.piece)) return 'Choose a piece from your collection.';
  if (![move.x,move.y,move.rotation].every(Number.isInteger) || move.rotation<0 || move.rotation>3 || typeof move.flipped!=='boolean') return 'Invalid placement.';
  const cells = shape(move.piece,move.rotation,move.flipped).map(([x,y])=>[x+move.x,y+move.y]);
  if (cells.some(([x,y])=>x<0 || y<0 || x>=SIZE || y>=SIZE)) return 'Keep all five squares on the board.';
  if (cells.some(([x,y])=>game.board[y*SIZE+x]!==null)) return 'That space is already taken.';
  if (game.remaining[game.turn].length===12) {
    const start = game.turn===0 ? 0 : SIZE-1;
    return cells.some(([x,y])=>x===start && y===start) ? null : 'Your first piece must cover your marked corner.';
  }
  const own = (x:number,y:number)=> x>=0 && y>=0 && x<SIZE && y<SIZE && game.board[y*SIZE+x]===game.turn;
  if (cells.some(([x,y])=>[[1,0],[-1,0],[0,1],[0,-1]].some(([dx,dy])=>own(x+dx,y+dy)))) return 'Your pieces can touch at corners, but never along an edge.';
  if (!cells.some(([x,y])=>[[1,1],[-1,1],[1,-1],[-1,-1]].some(([dx,dy])=>own(x+dx,y+dy)))) return 'Connect to a corner of one of your pieces.';
  return null;
}
export function findMove(game: Game, preferred?: string): Move | null {
  if (game.finished) return null;
  const pieces = preferred ? game.remaining[game.turn].filter(p=>p===preferred) : game.remaining[game.turn];
  for (const piece of pieces) {
    const seen = new Set<string>();
    for (const flipped of [false,true]) for (let rotation=0;rotation<4;rotation++) {
      const key=JSON.stringify(shape(piece,rotation,flipped).map(c=>c.join(',')).sort());
      if (seen.has(key)) continue; seen.add(key);
      for (let y=0;y<SIZE;y++) for(let x=0;x<SIZE;x++) {
        const move={piece,rotation,flipped,x,y}; if (!validate(game,move)) return move;
      }
    }
  }
  return null;
}
export function play(game: Game, move: Move): Game {
  const error=validate(game,move); if(error) throw new Error(error);
  const next=structuredClone(game);
  next.lastMove=shape(move.piece,move.rotation,move.flipped).map(([x,y])=>(y+move.y)*SIZE+x+move.x);
  next.lastMove.forEach(i=>next.board[i]=game.turn);
  next.remaining[game.turn]=next.remaining[game.turn].filter(p=>p!==move.piece);
  next.history.push({player: game.turn,piece:move.piece,squares:5});
  next.moves++; next.passes=0; next.turn=game.turn===0?1:0;
  if (!next.remaining[0].length && !next.remaining[1].length) next.finished=true;
  return next;
}
export function pass(game: Game): Game {
  if (game.finished) throw new Error('This game is complete.');
  if (findMove(game)) throw new Error('You still have a legal move. Try the hint button.');
  return {...game,turn:game.turn===0?1:0,passes:game.passes+1,finished:game.passes>=1,lastMove:[]};
}
export function score(game: Game, player: Player) { return (12-game.remaining[player].length)*5; }
