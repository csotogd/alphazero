"""Loopback-only training studio. Starts bounded local jobs; never cloud jobs."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from threading import Lock
import argparse,json,os,signal,subprocess,sys,time
from .train import ROOT,Config,atomic_json

RUNS=ROOT/'runs';WEB=ROOT/'web'
class Studio:
    def __init__(self):self.process=None;self.run_name=None;self.lock=Lock();self.log=None
    def running(self):return self.process is not None and self.process.poll() is None
    def directory(self,name):
        if not name or not all(c.isalnum() or c in '-_' for c in name):raise ValueError('Invalid run name')
        directory=RUNS/name
        if not directory.is_dir():raise ValueError('Run not found')
        return directory
    def list_runs(self):
        result=[]
        for p in sorted(RUNS.glob('*/status.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            try:
                data=json.loads(p.read_text());result.append({'name':p.parent.name,'game':data['game'],'status':data['status'],'iteration':data['iteration'],'games':data['games_played']})
            except (ValueError,KeyError):continue
        return result
    def status(self,name=None):
        runs=self.list_runs();name=name or self.run_name or (runs[0]['name'] if runs else None)
        data={}
        if name:
            directory=self.directory(name)
            if (directory/'status.json').exists():data=json.loads((directory/'status.json').read_text())
            if self.run_name==name and self.process and self.process.poll() is not None and data.get('status')=='running':data.update(status='failed',error='Trainer exited unexpectedly. See train.log; last checkpoint remains available.')
        return {'runs':runs,'selected':name,'active':self.run_name if self.running() else None,'training':data}
    def start(self,body):
        with self.lock:
            if self.running():raise ValueError('A training run is already active. Pause it first.')
            if body.get('resume'):
                directory=self.directory(body['run']);config=Config(**json.loads((directory/'config.json').read_text()))
                status=json.loads((directory/'status.json').read_text())
                if status.get('status')=='complete':config.iterations+=int(body.get('iterations',10))
                atomic_json(directory/'config.json',config.__dict__)
            else:
                config=Config(game=body.get('game','pentomino'),iterations=int(body.get('iterations',10)),games_per_iteration=int(body.get('games_per_iteration',4)),simulations=int(body.get('simulations',64)))
                if config.game not in ('pentomino','tictactoe'):raise ValueError('Unknown game')
                if not 1<=config.iterations<=2000 or not 1<=config.games_per_iteration<=64 or not 1<=config.simulations<=1600:raise ValueError('Settings outside supported local limits')
                if config.game=='tictactoe':config.channels=32;config.blocks=2;config.dirichlet_alpha=.3
                directory=RUNS/f'{config.game}-{time.strftime("%Y%m%d-%H%M%S")}'
                directory.mkdir(parents=True,exist_ok=False);atomic_json(directory/'config.json',config.__dict__)
            command=[sys.executable,'-m','az.train','--config',str(directory/'config.json'),'--run-dir',str(directory)]
            if body.get('resume'):command.append('--resume')
            if self.log:self.log.close()
            self.log=(directory/'train.log').open('a');self.process=subprocess.Popen(command,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            self.run_name=directory.name
            return {'run':directory.name,'pid':self.process.pid}
    def pause(self):
        with self.lock:
            if not self.running():raise ValueError('No active trainer')
            self.process.send_signal(signal.SIGINT)
            return {'status':'pausing','run':self.run_name}

studio=Studio()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_json(self,data,status=200):
        payload=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        parsed=urlparse(self.path)
        try:
            if parsed.path=='/api/status':return self.send_json(studio.status(parse_qs(parsed.query).get('run',[None])[0]))
            if parsed.path=='/api/replay':
                directory=studio.directory(parse_qs(parsed.query).get('run',[''])[0]);p=directory/'latest-game.json'
                return self.send_json(json.loads(p.read_text()) if p.exists() else {})
            files={'/':'index.html','/studio.css':'studio.css','/studio.js':'studio.js'}
            if parsed.path not in files:return self.send_json({'error':'Not found'},404)
            p=WEB/files[parsed.path];payload=p.read_bytes();self.send_response(200);self.send_header('Content-Type',{'html':'text/html; charset=utf-8','css':'text/css','js':'text/javascript'}[p.suffix[1:]]);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        except Exception as e:self.send_json({'error':str(e)},400)
    def do_POST(self):
        try:
            host=self.headers.get('Host','');origin=self.headers.get('Origin')
            if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):return self.send_json({'error':'Invalid host'},403)
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):return self.send_json({'error':'Invalid origin'},403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send_json({'error':'JSON required'},415)
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=4096:return self.send_json({'error':'Invalid request length'},400)
            body=json.loads(self.rfile.read(length))
            if self.path=='/api/start':return self.send_json(studio.start(body))
            if self.path=='/api/pause':return self.send_json(studio.pause())
            return self.send_json({'error':'Not found'},404)
        except Exception as e:self.send_json({'error':str(e)},400)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8787);args=parser.parse_args()
    RUNS.mkdir(exist_ok=True);server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'AlphaZero Studio: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if studio.running():studio.pause();studio.process.wait(timeout=30)
        server.server_close()
if __name__=='__main__':main()
