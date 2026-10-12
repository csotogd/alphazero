'use client';
import {useCallback,useEffect,useRef,useState} from 'react';
import {ArrowUpRight, ArrowRight, RotateCw, FlipHorizontal2, Sparkles, HelpCircle, X, Copy, Check, Users, Globe2, ChevronRight, Flag, Volume2, VolumeX, Circle, Plus, CornerDownLeft, MousePointer2} from 'lucide-react';
import {SIZE,PIECES,newGame,shape,validate,play,pass,score,findMove,type Game,type Move,type Player} from '@/lib/game/engine';
import {roomRequest,readRoom,type Session,type RoomView} from '@/lib/game/room-client';
import {Piece} from './piece';

type Dialog='rules'|'online'|'new'|null;
const DEFAULT_NAMES:[string,string]=['Violet','Lime'];
export function GameApp(){
 const [game,setGame]=useState<Game>(newGame);
 const [selected,setSelected]=useState('P');
 const [rotation,setRotation]=useState(0),[flipped,setFlipped]=useState(false);
 const [anchor,setAnchor]=useState<[number,number]|null>(null),[hover,setHover]=useState<[number,number]|null>(null);
 const [dialog,setDialog]=useState<Dialog>(null),[notice,setNotice]=useState('');
 const [room,setRoom]=useState<RoomView|null>(null),[session,setSession]=useState<Session|null>(null);
 const [busy,setBusy]=useState(false),[networkError,setNetworkError]=useState('');
 const [name,setName]=useState(''),[code,setCode]=useState(''),[tab,setTab]=useState<'create'|'join'>('create');
 const [sound,setSound]=useState(false),[copied,setCopied]=useState(false),[loaded,setLoaded]=useState(false);
 const dialogRef=useRef<HTMLDialogElement>(null),revisionRef=useRef(-1),syncRef=useRef(0);
 const names=room?.names||DEFAULT_NAMES;
 const myTurn=!room || (room.ready && room.player===game.turn);
 const editable=(!session || !!room) && myTurn && !game.finished && !busy && !networkError;
 const playerColor=game.turn===0?'violet':'lime';
 const point=anchor||hover;
 const move:Move|null=point?{piece:selected,rotation,flipped,x:point[0],y:point[1]}:null;
 const error=move?validate(game,move):null;
 const preview=new Set(move?shape(selected,rotation,flipped).map(([x,y])=>[x+move.x,y+move.y]).filter(([x,y])=>x>=0 && y>=0 && x<SIZE && y<SIZE).map(([x,y])=>y*SIZE+x):[]);
 const resetSelection=useCallback(()=>{setAnchor(null);setHover(null);setRotation(0);setFlipped(false);setNotice('');},[]);
 const applyRoom=useCallback((next:RoomView)=>{
   if(next.revision<revisionRef.current)return;
   if(next.revision!==revisionRef.current){setGame(next.game);resetSelection();}
   revisionRef.current=next.revision;setRoom(next);setNetworkError('');
 },[resetSelection]);
 useEffect(()=>{
   const invite=new URLSearchParams(window.location.search).get('room');
   try {
     const saved=localStorage.getItem('pentomino-local-v1');
     if(saved){const parsed=JSON.parse(saved);if(parsed.board?.length===SIZE*SIZE && parsed.remaining?.length===2 && [0,1].includes(parsed.turn))setGame(parsed);}
     const savedRoom=sessionStorage.getItem('pentomino-room-v1');
     if(savedRoom){const parsed=JSON.parse(savedRoom);if(parsed.code && parsed.token && (!invite || invite===parsed.code))setSession(parsed);}
   }catch{}
   if(invite){setCode(invite.toUpperCase());setTab('join');setDialog('online');}
   setLoaded(true);
 },[]);
 useEffect(()=>{if(loaded && !room && !session)try{localStorage.setItem('pentomino-local-v1',JSON.stringify(game));}catch{}},[game,room,session,loaded]);
 useEffect(()=>{
   if(!session)return;
   let active=true;let timer:ReturnType<typeof setTimeout>;const controller=new AbortController();const generation=++syncRef.current;
   const poll=async()=>{try{const next=await readRoom(session,controller.signal);if(active && generation===syncRef.current){const restoring=revisionRef.current<0;applyRoom(next);if(restoring)setDialog(d=>d==='online'?null:d);}}catch(e){if(active)setNetworkError(e instanceof Error?e.message:'Reconnecting…');}finally{if(active)timer=setTimeout(poll,1800);}};
   void poll();return()=>{active=false;controller.abort();clearTimeout(timer);};
 },[session,applyRoom]);
 useEffect(()=>{if(!game.remaining[game.turn].includes(selected))setSelected(game.remaining[game.turn][0]||'P');},[game,selected]);
 useEffect(()=>{if(dialog && !dialogRef.current?.open)dialogRef.current?.showModal();else if(!dialog && dialogRef.current?.open)dialogRef.current?.close();},[dialog]);
 function chime(){if(!sound)return;try{const ctx=new AudioContext();const osc=ctx.createOscillator(),gain=ctx.createGain();osc.connect(gain);gain.connect(ctx.destination);osc.frequency.setValueAtTime(game.turn===0?440:554.37,ctx.currentTime);gain.gain.setValueAtTime(.04,ctx.currentTime);gain.gain.exponentialRampToValueAtTime(.001,ctx.currentTime+.18);osc.start();osc.stop(ctx.currentTime+.2);osc.onended=()=>void ctx.close();}catch{}}
 async function commit(){
   if(!move || error || !editable)return;setBusy(true);setNotice('');
   try{if(session && room){applyRoom(await roomRequest({action:'move',code:session.code,revision:room.revision,move},session));}else{setGame(play(game,move));resetSelection();}chime();}catch(e){setNotice((e as Error).message);}finally{setBusy(false);}
 }
 useEffect(()=>{
   const key=(event:KeyboardEvent)=>{
     if(dialog || !editable || (event.target instanceof HTMLElement && ['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)))return;
     if(event.key.toLowerCase()==='r'){event.preventDefault();setRotation(r=>(r+1)%4);}
     if(event.key.toLowerCase()==='f'){event.preventDefault();setFlipped(f=>!f);}
     if(event.key==='Escape')setAnchor(null);
     if(event.key==='Enter' && anchor && !(event.target instanceof HTMLButtonElement)){event.preventDefault();void commit();}
   };window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
 });
 async function connect(){setBusy(true);setNotice('');try{
   const next=await roomRequest({action:tab,name,code:code.trim().toUpperCase()});
   const connection={code:next.code,token:next.token!};revisionRef.current=-1;applyRoom(next);setSession(connection);sessionStorage.setItem('pentomino-room-v1',JSON.stringify(connection));setDialog(null);
   window.history.replaceState(null,'',`?room=${next.code}`);
 }catch(e){setNotice((e as Error).message);}finally{setBusy(false);}}
 function localGame(){syncRef.current++;setSession(null);setRoom(null);revisionRef.current=-1;setNetworkError('');sessionStorage.removeItem('pentomino-room-v1');window.history.replaceState(null,'',window.location.pathname);setGame(newGame());resetSelection();setDialog(null);}
 async function skip(){if(!editable)return;setBusy(true);try{if(session && room)applyRoom(await roomRequest({action:'pass',code:session.code,revision:room.revision},session));else {setGame(pass(game));resetSelection();}}catch(e){setNotice((e as Error).message);}finally{setBusy(false);}}
 async function rematch(){if(!session || !room){localGame();return;}setBusy(true);try{applyRoom(await roomRequest({action:'rematch',code:session.code,revision:room.revision},session));}catch(e){setNotice((e as Error).message);}finally{setBusy(false);}}
 function hint(){const next=findMove(game,selected)||findMove(game);if(next){setSelected(next.piece);setRotation(next.rotation);setFlipped(next.flipped);setAnchor([next.x,next.y]);setNotice('A little nudge. Place this piece, or find your own path.');}else setNotice('No legal moves left. Pass your turn to continue.');}
 async function copyInvite(){if(!room)return;try{await navigator.clipboard.writeText(`${window.location.origin}?room=${room.code}`);setCopied(true);setTimeout(()=>setCopied(false),2200);}catch{setNotice(`Share this room code with your friend: ${room.code}`);}}
 const title=game.finished?(score(game,0)===score(game,1)?'A perfectly matched pair.':`${names[score(game,0)>score(game,1)?0:1]} takes the board.`):room && !room.ready?'Good things come in twos.':`${names[game.turn]}’s turn`;
 return <div className="site-shell">
  <header className="topbar">
   <a className="wordmark" href="/" aria-label="Pentomino home"><span className="brand-piece"><Piece piece="P"/></span>pentomino<span className="brand-dot">✳</span></a>
   <nav aria-label="Main navigation"><span className="nav-current">The game</span><button onClick={()=>setDialog('rules')}>How to play <ArrowUpRight size={14}/></button></nav>
   <button className="invite-button" onClick={()=>{setNotice('');setDialog('online');}}><Users size={16}/> Play with a friend <ArrowUpRight size={16}/></button>
  </header>
  <main>
   <section className="intro"><div><div className="eyebrow"><span/> A LITTLE SPACE FOR STRATEGY</div><h1>Small pieces. <em>Big possibilities.</em></h1><p>Find your corner. Make your move. Leave your mark.</p></div><div className="intro-mark" aria-hidden="true"><Piece piece="F"/><span>12 pieces.<br/>Endless ways to play.</span></div></section>
   <section className="game-area" aria-label="Pentomino game">
    <div className="game-toolbar"><div className="mode-switch"><button className={!room?'active':''} onClick={()=>room?setDialog('new'):null}><Users size={15}/> Side by side</button><button className={room?'active':''} onClick={()=>{setNotice('');setDialog('online');}}><Globe2 size={15}/> Online</button></div><div className="game-meta"><span className="live-dot"/>{room?`ROOM ${room.code}`:'LOCAL GAME'}<span className="meta-divider"/><button onClick={()=>{setNotice('');setDialog('new');}}><Plus size={14}/> New game</button></div></div>
    {room && <div className="room-banner"><span><Globe2 size={16}/>{room.ready?`You’re playing as ${names[room.player]}. ${networkError?'Reconnecting…':'Room connected.'}`:'Your room is ready. Send the invitation to your friend.'}</span><button onClick={copyInvite}>{copied?<Check size={14}/>:<Copy size={14}/>} {copied?'Copied!':'Copy invite'} <b>{room.code}</b></button></div>}
    <div className="play-layout">
     <div className="board-column">
      <div className="players">{([0,1] as Player[]).map(p=><div key={p} className={`player-card ${p===0?'violet':'lime'} ${game.turn===p && !game.finished?'is-turn':''}`}><div className="player-avatar"><Piece piece={p===0?'P':'W'}/></div><div className="player-info"><span className="player-label">PLAYER 0{p+1}{room?.player===p?' · YOU':''}</span><strong>{names[p]}</strong></div><div className="player-score"><b>{score(game,p).toString().padStart(2,'0')}</b><span>points</span></div>{game.turn===p && !game.finished && <span className="turn-dot"/>}</div>)}</div>
      <div className={`board-frame ${playerColor}`}>
       <div className="board-top-label"><span>THE PLAYING FIELD</span><span>12 × 12</span></div>
       <div className="board-wrap"><div className="coordinates top-coordinates">{Array.from({length:SIZE},(_,i)=><span key={i}>{String.fromCharCode(65+i)}</span>)}</div><div className="coordinates side-coordinates">{Array.from({length:SIZE},(_,i)=><span key={i}>{i+1}</span>)}</div>
        <div className="board" role="group" aria-label="12 by 12 game board" onMouseLeave={()=>setHover(null)}>{game.board.map((cell,i)=>{
          const x=i%SIZE,y=Math.floor(i/SIZE),isPreview=editable && preview.has(i),start=i===0?0:i===SIZE*SIZE-1?1:null;
          return <button key={i} aria-label={`${String.fromCharCode(65+x)}${y+1}${cell!==null?`, ${names[cell]} piece`:start!==null?`, ${DEFAULT_NAMES[start]} starting corner`:''}`} aria-pressed={isPreview} className={`board-cell ${cell===0?'filled cell-violet':cell===1?'filled cell-lime':''} ${isPreview?(error?'preview invalid':`preview preview-${playerColor}`):''} ${game.lastMove.includes(i)?'last-move':''} ${cell===null && start!==null?`starting start-${start}`:''}`} onMouseEnter={()=>{if(editable)setHover([x,y]);}} onFocus={()=>{if(editable)setHover([x,y]);}} onClick={()=>{if(editable){setAnchor([x,y]);setNotice('');}}} disabled={!editable}>{cell===null && start!==null && !isPreview && <span>✳</span>}</button>;
        })}</div>
       </div>
       <div className="board-bottom-label"><span><span className="tiny-dot violet-dot"/> Violet starts at A1</span><span>Lime starts at L12 <span className="tiny-dot lime-dot"/></span></div>
      </div>
      <div className="board-footnote"><span><MousePointer2 size={13}/> Pick a piece. Tap the board. Make it yours.</span><button onClick={()=>setSound(!sound)} aria-label={sound?'Mute game sounds':'Enable game sounds'} title={sound?'Sound on':'Sound off'}>{sound?<Volume2 size={16}/>:<VolumeX size={16}/>}</button></div>
     </div>
     <aside className={`piece-panel ${playerColor}`}>
      <div className="turn-heading"><span className={`tiny-dot ${playerColor}-dot`}/><span>{game.finished?'THE FINAL SCORE':room && !room.ready?'WAITING ROOM':`TURN ${String(game.moves+1).padStart(2,'0')}`}</span><span className="turn-status">{game.finished?'Well played':myTurn?'Let’s play':'Take a breath'}</span></div>
      <h2>{title}</h2><p className="turn-description">{game.finished?'Every little piece made a difference. Ready for another round?':room && !room.ready?'Copy your invite above. Your game begins when your friend joins.':myTurn?game.remaining[game.turn].length===12?'Start something good. Cover your marked corner with your first piece.':'Keep growing. Touch your own pieces at the corners, never the edges.':`${names[game.turn]} is finding their next move. Your pieces will appear when it’s your turn.`}</p>
      {game.finished?<div className="end-card"><Sparkles size={32}/><h3>{Math.max(score(game,0),score(game,1))} squares of possibility.</h3><p>{names[0]} {score(game,0)} <span>—</span> {score(game,1)} {names[1]}</p><button className="primary" disabled={busy || !!(room && (room.rematch & (1<<room.player)))} onClick={rematch}>{room && (room.rematch & (1<<room.player))?'Waiting for opponent…':'Play another round'} <ArrowRight size={17}/></button>{room && room.rematch!==0 && !(room.rematch & (1<<room.player)) && <p>Your opponent wants a rematch.</p>}</div>:<>
       <div className="collection-heading"><span>YOUR COLLECTION</span><span>{game.remaining[game.turn].length} <span>/ 12 pieces</span></span></div>
       <div className="piece-grid">{Object.keys(PIECES).map(piece=>{const used=!game.remaining[game.turn].includes(piece);return <button key={piece} className={`piece-option ${selected===piece && !used?'selected':''} ${used?'used':''}`} disabled={used || !editable} aria-label={`${piece} pentomino${used?', already placed':''}`} aria-pressed={selected===piece && !used} onClick={()=>{setSelected(piece);resetSelection();}}><span className="piece-name">{piece}</span><Piece piece={piece}/>{used?<Check className="piece-check" size={13}/>:selected===piece?<span className="selected-dot"/>:null}</button>;})}</div>
       <div className="transform-controls"><button disabled={!editable} onClick={()=>setRotation(r=>(r+1)%4)}><RotateCw size={16}/> Rotate <kbd>R</kbd></button><button disabled={!editable} onClick={()=>setFlipped(f=>!f)}><FlipHorizontal2 size={16}/> Flip <kbd>F</kbd></button></div>
       <div className="placement-preview"><span className="selected-mini"><Piece piece={selected} rotation={rotation} flipped={flipped}/></span><span>{anchor?error?'Find another spot':'Looking good. Make your move.':`${selected} piece selected`}<small>{anchor?error?'Rotate, flip, or try another square.':`Ready at ${String.fromCharCode(65+anchor[0])}${anchor[1]+1}`:'Tap a square to preview your placement.'}</small></span>{anchor && <button aria-label="Clear placement" onClick={()=>setAnchor(null)}><X size={15}/></button>}</div>
       <button className="primary place-button" onClick={commit} disabled={!anchor || !!error || !editable}>{busy?'Placing…':'Place piece'}<CornerDownLeft size={17}/></button>
       <div className="secondary-actions"><button disabled={!editable} onClick={hint}><Sparkles size={14}/> A little hint</button><button disabled={!editable} onClick={skip}>Pass turn <ChevronRight size={14}/></button></div>
      </>}
      <div className="feedback" role="status" aria-live="polite">{networkError || notice || (anchor && error?error:'')}</div>
     </aside>
    </div>
   </section>
   <section className="principles" aria-label="The game in three steps"><div><span className="principle-number">01</span><div><h3>Five squares. One piece.</h3><p>Twelve unique shapes. All yours to play.</p></div><span className="principle-art"><Piece piece="T"/></span></div><div><span className="principle-number">02</span><div><h3>Make a little connection.</h3><p>Your pieces meet at corners, not edges.</p></div><span className="corner-art" aria-hidden="true"><i/><i/></span></div><div><span className="principle-number">03</span><div><h3>Claim your kind of space.</h3><p>Place the most squares to take the win.</p></div><Flag size={25} strokeWidth={1.2}/></div></section>
   <footer><span>Made for a little friendly competition.</span><span>NO RUSH. JUST YOUR NEXT MOVE. <span className="footer-star">✳</span></span></footer>
  </main>
  <dialog ref={dialogRef} className="modal" onCancel={()=>setDialog(null)} onClick={e=>{if(e.target===e.currentTarget)setDialog(null);}}><div className="modal-inner"><button className="modal-close" aria-label="Close dialog" onClick={()=>{setDialog(null);setNotice('');}}><X size={20}/></button>
   {dialog==='rules' && <><div className="eyebrow">A SMALL GUIDE</div><h2>A good game starts<br/>with a little connection.</h2><p>Pentomino is a territory game for two. Each player gets the same twelve shapes, each made of five squares.</p><ol className="rules-list"><li><b>Find your starting corner.</b><span>Violet starts at A1, Lime at L12. Your first piece must cover that square.</span></li><li><b>Grow from the corners.</b><span>Each new piece must touch one of your pieces diagonally. Your own pieces may never share an edge. Opposing pieces can touch anywhere.</span></li><li><b>Make each piece count.</b><span>Rotate and flip any piece. Keep it inside the board, with no overlaps. Each placed piece earns five points.</span></li><li><b>Know when to pass.</b><span>You can pass only when no legal moves remain. Two consecutive passes end the game. Most points wins; equal scores are a draw.</span></li></ol><button className="primary" onClick={()=>setDialog(null)}>I’ve got this <ArrowRight size={16}/></button><p className="modal-note">Keyboard: R to rotate · F to flip · Enter to place · Esc to clear</p></>}
   {dialog==='online' && <><div className="eyebrow">BETTER TOGETHER</div><h2>A little friendly<br/>competition.</h2><p>Create a room and send the invite. Or join a friend with their six-character code.</p>{room?<div className="existing-room"><p>You’re in room <b>{room.code}</b> as {names[room.player]}.</p><button className="primary" onClick={copyInvite}>{copied?'Copied!':'Copy invitation'}<Copy size={16}/></button><button className="text-button" onClick={()=>setDialog('new')}>Leave room and play locally</button></div>:<><div className="dialog-tabs"><button className={tab==='create'?'active':''} onClick={()=>{setTab('create');setNotice('');}}>Create a room</button><button className={tab==='join'?'active':''} onClick={()=>{setTab('join');setNotice('');}}>Join a room</button></div><form onSubmit={e=>{e.preventDefault();void connect();}}><label>Your name<input autoComplete="nickname" maxLength={24} placeholder="What should we call you?" value={name} onChange={e=>setName(e.target.value)} required/></label>{tab==='join' && <label>Room code<input className="code-input" placeholder="ABC123" value={code} onChange={e=>setCode(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g,''))} minLength={6} maxLength={6} required/></label>}<button className="primary" disabled={busy}>{busy?'Connecting…':tab==='create'?'Create my room':'Join the game'}<ArrowRight size={17}/></button></form></>}<p className="dialog-error" role="status">{notice}</p><p className="modal-note">Two players · No timers · Rooms stay available for 7 days</p></>}
   {dialog==='new' && <><div className="eyebrow">A FRESH PERSPECTIVE</div><h2>Make room<br/>for a new game.</h2><p>{room?'Leave this room and start a new game on this device. Your opponent’s room will remain available.':'Start a fresh board for two players sharing this device. Your current local game will be replaced.'}</p><button className="primary" onClick={localGame}>Start a local game <ArrowRight size={17}/></button><button className="text-button" onClick={()=>{setNotice('');setDialog('online');}}>Or play with a friend online <ArrowUpRight size={15}/></button></>}
  </div></dialog>
 </div>;
}
