"""Render the completed diagnostic evidence as a standalone local HTML report."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();study=Path(args.study);output=Path(args.output)
    matches=[json.loads(line) for line in (study/'matches.jsonl').read_text().splitlines()]
    by_name={r['name']:r for r in matches}
    audit=json.loads((study/'audit.json').read_text())
    comparisons=json.loads((study/'comparisons.json').read_text())
    protocol=json.loads((study/'protocol.json').read_text())
    selected=json.loads((study/'selection.json').read_text())['selected_training_variant']
    endgames=json.loads((study/'endgames.json').read_text())
    source_games=sum(window['games'] for window in audit['selfplay_windows'])
    full_inventory_draws=round(sum(window['games']*window['full_inventory_draw_rate'] for window in audit['selfplay_windows']))
    blind_spots=sum(r['has_suboptimal_actions'] and r['chosen_exact_values']['64']<r['optimal_value']
                   and r['optimal_action_visits']['64']==r['optimal_action_visits']['256']==0
                   for r in endgames['rows'])
    e=html.escape
    grouped=lambda number: format(number, ",").replace(",", ".")
    names={'last_64':'Modelo original · 64 simulaciones','last_policy_only':'Política sin búsqueda',
           'last_16':'16 simulaciones','last_256':'256 simulaciones',
           'last_explore5':'Más exploración PUCT (5 frente a 1,25)',
           'last_zero_value':'Valor neuronal sustituido por cero',
           'last_uniform_prior':'Prior uniforme; conserva valor neuronal',
           'last_transposed':'Red sobre tablero reflejado',
           'last_symmetry_ensemble':'Promedio de red original y reflejada',
           'random_vs_random':'Aleatorio contra aleatorio','frontier_vs_random':'Heurística contra aleatorio',
           'confirm_last64':'Confirmación · original 64','confirm_last256':'Confirmación · original 256',
           'last1000':'Original · 1.000 simulaciones'}
    def label(name):
        if name in names:return names[name]
        return name.replace('checkpoint_','Modelo tras ').replace('_seed',' · semilla ').replace('_',' ')
    def table(rows):
        body=''.join(f'<tr><td>{e(label(r["name"]))}</td><td>{r["games"]}</td>'
                     f'<td>{r["wins"]} / {r["draws"]} / {r["losses"]}</td>'
                     f'<td><strong>{100*r["win_rate"]:.1f}%</strong></td>'
                     f'<td>{100*r["score_rate"]:.1f}%</td>'
                     f'<td>{r["mean_candidate_points"]:.1f} – {r["mean_opponent_points"]:.1f}</td></tr>' for r in rows)
        return '<div class="scroll"><table><thead><tr><th>Variante</th><th>Partidas</th><th>G / E / P</th><th>Victorias</th><th>Puntuación</th><th>Puntos medios</th></tr></thead><tbody>'+body+'</tbody></table></div>'
    def delta(name):
        row=comparisons[name];lo,hi=row['win_rate_delta_ci99']
        return f'{100*row["win_rate_delta"]:+.1f} puntos porcentuales; intervalo exploratorio del 99% [{100*lo:+.1f}, {100*hi:+.1f}]'
    total=sum(r['games'] for r in matches)
    training_games=protocol['continuation_games_per_arm']*10
    c64,c256=by_name['confirm_last64'],by_name['confirm_last256']
    search1000_bundle=None
    search1000_html=''
    search1000_dir=study/'search1000'
    if (search1000_dir/'comparison.json').exists():
        result=json.loads((search1000_dir/'matches.jsonl').read_text().splitlines()[-1])
        differences=json.loads((search1000_dir/'comparison.json').read_text())
        timing=json.loads((search1000_dir/'timing.json').read_text())
        search1000_bundle={'result':result,'comparisons':differences,'timing':timing,
                          'protocol':json.loads((search1000_dir/'protocol.json').read_text())}
        total+=result['games']
        search1000_html='<h2>Seguimiento solicitado: 1.000 simulaciones</h2>'
        search1000_html+='<p>Misma red, sin reentrenar, frente al mismo rival aleatorio y con las mismas semillas y colores de las comparaciones de 64 y 256 simulaciones. Se decidió después de conocer sus resultados: es un seguimiento adaptativo, no una nueva confirmación independiente.</p>'
        search1000_html+=table([c64,c256,result])
        first_wins=result['by_color']['0'].get('1.0',0)
        second_wins=result['by_color']['1'].get('1.0',0)
        search1000_html+=f'<p><strong>Por orden de turno:</strong> con 1.000 simulaciones gana {first_wins}/100 empezando primero y {second_wins}/100 jugando segundo. Es un resultado de esta muestra, no una garantía del 99% ni una demostración del valor teórico de cada turno.</p>'
        search1000_html+='<ul>'+''.join(f'<li>Frente a {e(label(name))}: {100*d["win_rate_delta"]:+.1f} puntos de victorias; intervalo exploratorio del 99% [{100*d["win_rate_delta_ci99"][0]:+.1f}, {100*d["win_rate_delta_ci99"][1]:+.1f}].</li>' for name,d in differences.items())+'</ul>'
        search1000_html+=f'<p>Coste medido: {timing["mean_seconds_per_move"]:.3f} segundos por jugada de media; percentil 90: {timing["p90_seconds_per_move"]:.3f} segundos. Evaluación completa: {result["seconds"]/60:.1f} minutos. El tiempo depende de las posiciones y del estado del ordenador; 1.000 simulaciones requieren 3,91 veces el presupuesto de 256, pero no necesariamente 3,91 veces el tiempo.</p>'
    prior_dir=study/'prior-followup'
    capacity_path=study/'replay-capacity'/'matches.jsonl'
    capacity_bundle=None
    capacity_confirmation=None
    capacity_html=''
    if capacity_path.exists():
        from .diagnose import paired_delta
        capacity_results=[json.loads(line) for line in capacity_path.read_text().splitlines()]
        capacity_comparisons={}
        for result in capacity_results:
            seed=result['name'].split('_seed')[-1]
            capacity_comparisons[seed]=paired_delta(result['outcomes'],by_name[f'control_seed{seed}']['outcomes'])
        capacity_bundle={'matches':capacity_results,'comparisons':capacity_comparisons,
                         'protocol':json.loads((capacity_path.parent/'protocol.json').read_text())}
        total+=sum(r['games'] for r in capacity_results)
        training_games+=80*len(capacity_results)
        capacity_html=f'<p><strong>Comparación adicional que sí aísla el descarte de la memoria:</strong> mismas posiciones iniciales, semillas, optimizador y learning rate, pero límite de 20.000 posiciones. En estas 80 partidas solo se llega a unas 6.500: se prueba conservar los ejemplos más antiguos, no el efecto de una memoria llena de 20.000. Son dos repeticiones exploratorias.</p>{table(capacity_results)}'
    capacity_confirm_path=study/'replay-capacity-confirmation'/'matches.jsonl'
    if capacity_confirm_path.exists():
        from .diagnose import paired_delta
        results=[json.loads(line) for line in capacity_confirm_path.read_text().splitlines()]
        differences={}
        for result in results:
            seed=result['name'].split('_seed')[-1]
            differences[seed]=paired_delta(result['outcomes'],by_name[f'confirm_control_seed{seed}']['outcomes'])
        capacity_confirmation={'matches':results,'comparisons':differences,
                 'protocol':json.loads((capacity_confirm_path.parent/'protocol.json').read_text())}
        total+=sum(r['games'] for r in results)
        capacity_html+='<p>La retención de memoria mejoró en ambas semillas durante la exploración y pasó a una confirmación adaptativa de 200 partidas nuevas por semilla:</p>'+table(results)
        capacity_html+='<ul>'+''.join(f'<li>Semilla {seed}, diferencia en victorias frente al control: {100*d["win_rate_delta"]:+.1f} puntos; intervalo exploratorio del 99% [{100*d["win_rate_delta_ci99"][0]:+.1f}, {100*d["win_rate_delta_ci99"][1]:+.1f}].</li>' for seed,d in differences.items())+'</ul>'
    symmetry_path=study/'symmetry-confirmation'/'matches.jsonl'
    symmetry_bundle=None
    symmetry_html=''
    if symmetry_path.exists():
        from .diagnose import paired_delta
        symmetry_result=json.loads(symmetry_path.read_text().splitlines()[-1])
        symmetry_comparison=paired_delta(symmetry_result['outcomes'],c64['outcomes'])
        symmetry_bundle={'result':symmetry_result,'comparison':symmetry_comparison}
        total+=symmetry_result['games']
        lo,hi=symmetry_comparison['win_rate_delta_ci99']
        symmetry_html=f'<p><strong>Confirmación adicional con nuevas semillas:</strong> {symmetry_result["wins"]}/{symmetry_result["games"]} victorias frente a {c64["wins"]}/{c64["games"]} del original. Diferencia: {100*symmetry_comparison["win_rate_delta"]:+.1f} puntos; intervalo exploratorio del 99% [{100*lo:+.1f}, {100*hi:+.1f}]. Esta comprobación se añadió tras seleccionar la mejor variante sin entrenamiento en la exploración inicial.</p>'
    prior_html=''
    prior_bundle=None
    if (prior_dir/'comparison.json').exists():
        prior_matches=[json.loads(line) for line in (prior_dir/'matches.jsonl').read_text().splitlines()]
        prior_comparison=json.loads((prior_dir/'comparison.json').read_text())
        prior_bundle={'matches':prior_matches,'comparison':prior_comparison,
                      'protocol':json.loads((prior_dir/'protocol.json').read_text())}
        total+=sum(r['games'] for r in prior_matches)
        lo,hi=prior_comparison['win_rate_delta_ci99']
        prior_html=f'''<h2>Prueba adicional: recomendaciones demasiado concentradas</h2>
<p>En {blind_spots} de los finales fallidos, las acciones óptimas recibieron cero visitas tanto con 64 como con 256 simulaciones. Con un prior uniforme, los errores en los {endgames['decision_relevant_positions']} finales con decisiones relevantes bajaron a {endgames['suboptimal_choices']['uniform64']}. Esto motivó una prueba adicional: suavizar la política con su raíz cuadrada, o mezclar un 10% de probabilidades uniformes. Se conserva el valor aprendido y no se usa el solucionador exacto durante las partidas.</p>
{table(prior_matches)}<p>Se seleccionó <strong>{e(prior_comparison['selected'])}</strong> en la exploración y se comparó con el modelo original en otras 200 partidas con nuevas semillas. Diferencia en victorias: {100*prior_comparison['win_rate_delta']:+.1f} puntos; intervalo exploratorio del 99% [{100*lo:+.1f}, {100*hi:+.1f}]. Esta prueba se añadió tras observar los finales, y se identifica como seguimiento adaptativo.</p>'''
    # A real, exactly solved counterexample, not an illustrative invented board.
    examples=[r for r in endgames['rows'] if r['legal']==4 and r['has_suboptimal_actions']
              and r['chosen_exact_values']['256']<r['optimal_value']]
    example_html=''
    entropy_path=study/'policy_entropy.json'
    entropy=json.loads(entropy_path.read_text()) if entropy_path.exists() else []
    entropy_html=''
    if entropy:
        entropy_html=f'<p>Una comprobación adicional sobre las mismas {entropy[0]["states"]} posiciones no muestra concentración global creciente: la entropía normalizada media pasó de {entropy[0]["mean_normalized_policy_entropy"]:.3f} a {entropy[-1]["mean_normalized_policy_entropy"]:.3f} entre los modelos de 1.000 y 6.000 partidas (1 = uniforme). Los fallos con probabilidades casi nulas son locales; no se debe inferir un colapso general de la política a partir de esos ejemplos.</p>'
    if examples:
        from .games.pentomino import Pentomino
        from .endgame_audit import solve_actions
        row=examples[0];game=Pentomino();state=game.from_web(row['state'])
        values,_=solve_actions(game,state)
        action=next(a for a,v in values.items() if v==max(values.values()))
        _,mask=game.by_action[action]
        cells=[]
        for i,owner in enumerate(row['state']['board']):
            color='#7465be' if owner==0 else '#d89050' if owner==1 else '#f3f2ed'
            if mask&(1<<i):color='#63b594'
            cells.append(f'<span style="background:{color};aspect-ratio:1;border-radius:2px"></span>')
        move=game.decode_action(action)
        example_html=f'''<div class="card"><strong>Un fallo reproducible con solo cuatro opciones</strong><div style="display:grid;grid-template-columns:repeat(12,1fr);gap:2px;max-width:330px;margin:18px 0">{''.join(cells)}</div><p>Morado y naranja: piezas ya colocadas. Verde: una colocación de la pieza <strong>{e(move['piece'])}</strong> que permite forzar la victoria. La red asignó en total un {100*row['optimal_prior_mass']:.3f}% a las acciones óptimas. MCTS no les dedicó ninguna visita y eligió una jugada cuyo mejor resultado es empatar, tanto con 64 como con 256 simulaciones.</p><small>Posición real de la tanda original. Resultado comprobado mediante minimax exacto, fuera del entrenamiento.</small></div>'''
    body=f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diagnóstico de AlphaZero · Pentomino</title><style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#253044;background:#f5f4ef;line-height:1.6}}body{{max-width:1120px;margin:0 auto;padding:40px 24px}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.15;letter-spacing:-1.5px}}h2{{font-size:23px;margin-top:38px}}p{{max-width:900px}}.eyebrow{{color:#6257a5;font-weight:650;letter-spacing:1px;font-size:12px;text-transform:uppercase}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:28px 0}}.card{{padding:20px;background:white;border:1px solid #e1dfd7;border-radius:16px}}.card b{{font-size:30px;display:block;color:#514591}}.card span{{color:#627080;font-size:14px}}.note{{background:#eeebfa;border-left:3px solid #7465be;padding:16px 20px;border-radius:6px}}.scroll{{overflow:auto;background:white;border-radius:12px;border:1px solid #e1dfd7}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:12px 14px;text-align:left;border-bottom:1px solid #eeedf0;white-space:nowrap}}th{{color:#617080;font-size:12px;text-transform:uppercase}}td:first-child{{white-space:normal;min-width:220px}}summary{{cursor:pointer;font-weight:650}}li{{margin:10px 0}}small{{color:#657080}}a{{color:#514591}}code{{font-size:12px;overflow-wrap:anywhere}}footer{{margin-top:40px;border-top:1px solid #ddd;padding-top:18px}}details{{margin:18px 0}}</style>
<div class="eyebrow">Experimento local · 13 octubre 2026</div><h1>¿Qué está limitando<br>el aprendizaje?</h1>
<p>Comparaciones controladas a partir del modelo guardado tras {grouped(source_games)} partidas. Se conserva el experimento original. Todas las cifras siguientes son mediciones, con sus límites indicados.</p>
<div class="cards"><div class="card"><b>{grouped(total)}</b><span>partidas de evaluación</span></div><div class="card"><b>{training_games}</b><span>nuevas partidas de autojuego</span></div><div class="card"><b>7</b><span>grupos de hipótesis</span></div><div class="card"><b>2</b><span>semillas por variante entrenada</span></div></div>
<div class="note"><strong>Cómo leerlo:</strong> “Victorias” cuenta solo partidas ganadas. “Puntuación” cuenta victoria = 1, empate = ½. Las pruebas iniciales sirven para explorar; la confirmación usa nuevas semillas. Ninguna variante demuestra aquí un 99% de victorias.</div>
<h2>Confirmación independiente</h2>
<p>Con el mismo modelo, 64 simulaciones obtuvieron <strong>{c64['wins']}/{c64['games']}</strong> victorias y 256 obtuvieron <strong>{c256['wins']}/{c256['games']}</strong>. Diferencia emparejada: {delta('confirm_search256_vs64')}.</p>
<p>Variante entrenada seleccionada entre las cuatro alternativas del protocolo inicial por la media de las dos semillas en la exploración: <strong>{e(label(selected))}</strong>. Se confirman ambas semillas, sin escoger la más afortunada. La comparación adicional del tamaño de memoria se documenta por separado.</p>
{table([r for r in matches if r['name'].startswith('confirm_')])}
<ul><li>Semilla 811, variante frente a continuación de control: {delta('confirm_seed811')}.</li><li>Semilla 977: {delta('confirm_seed977')}.</li></ul>
<h2>1. Learning rate</h2><p>La tanda original usó 0,02 y bajó a 0,002 alrededor de la partida 2.500. Probamos continuaciones con 0,0002, 0,002 y 0,02, conservando pesos iniciales, momento del optimizador, memoria inicial y presupuesto de entrenamiento. Son pruebas cortas de {protocol['continuation_games_per_arm']} partidas y {protocol['continuation_games_per_arm']*4} actualizaciones por variante; no establecen la tasa óptima para un entrenamiento completo.</p>
{table([r for r in matches if r['name'].startswith(('control_','lr_low_','lr_high_'))])}
<h2>2. Búsqueda y exploración</h2><p>En las posiciones auditadas, la mediana es {audit['legal_actions']['median']:.0f} jugadas legales y el percentil 90 es {audit['legal_actions']['p90']:.0f}. Con 64 simulaciones no siempre se exploran todas. Cambiar el presupuesto permite comprobar el efecto sin reentrenar. PUCT = 5 modifica cuánto se exploran alternativas.</p>
{table([by_name[k] for k in ['last_policy_only','last_16','last_64','last_256','last_explore5']])}
{search1000_html}
<h2>3. Memoria y diversidad del autojuego</h2><p>La memoria conserva {grouped(audit['replay_positions'])} posiciones, aproximadamente {audit['approx_replay_games']:.0f} partidas. Probamos repaso de resultados antiguos para la cabeza de valor y un ruido Dirichlet más repartido (0,3 frente a 0,03). No se conservaron las distribuciones MCTS antiguas fuera de la memoria actual, por lo que el repaso histórico no es una prueba pura de aumentar el tamaño de la memoria: añade ejemplos de valor y cambia la composición de los lotes.</p>
{table([r for r in matches if r['name'].startswith(('history_value_','noise_spread_'))])}
{capacity_html}
<h2>4. Calidad del valor y de la política</h2><p>Para separar funciones, anulamos el valor aprendido manteniendo la política, y sustituimos la política por probabilidades uniformes manteniendo el valor. Además resolvimos exactamente {endgames['solved']} finales pequeños: {endgames['decision_relevant_positions']} tenían decisiones con distinto resultado. La política sola escogió una acción inferior en {endgames['suboptimal_choices']['policy']}, MCTS-64 en {endgames['suboptimal_choices']['64']} y MCTS-256 en {endgames['suboptimal_choices']['256']}. Esta selección de finales no representa todas las posiciones.</p>
{table([by_name[k] for k in ['last_64','last_zero_value','last_uniform_prior']])}
{example_html}{entropy_html}{prior_html}
<h2>5. Simetría y generalización</h2><p>Reflejar el tablero por su diagonal conserva exactamente las reglas y ambos puntos de inicio. En {audit['diagonal_symmetry']['states']} posiciones, la diferencia absoluta media del valor fue {audit['diagonal_symmetry']['mean_abs_value_difference']:.3f} en una escala de −1 a +1. Comparamos el modelo reflejado y el promedio de ambas orientaciones. El promedio cuesta dos evaluaciones neuronales por nodo: no es una comparación a igual tiempo.</p>
{table([by_name[k] for k in ['last_64','last_transposed','last_symmetry_ensemble']])}
{symmetry_html}
<h2>6. Empates, colores y referencia aleatoria</h2><p>En {grouped(source_games)} partidas originales hubo {full_inventory_draws} empates con ambos jugadores habiendo colocado todas sus piezas. No demuestra que el 99% sea alcanzable ni establece un techo de juego perfecto. Se alternan colores; “aleatorio” selecciona uniformemente entre las colocaciones legales, no entre tipos de pieza.</p>
{table([by_name[k] for k in ['random_vs_random','frontier_vs_random','last_64']])}
<h2>7. Implementación y evolución real</h2><p>Las pruebas verifican legalidad, equivalencia Python/TypeScript, alternancia de jugadores, signos de MCTS, resultados terminales, simetrías, persistencia y evaluación reproducible. Pasarlas descarta esos fallos concretos; no prueba ausencia de cualquier bug. Los modelos históricos se evalúan aquí con las mismas semillas para reducir el ruido de la comparación temporal.</p>
{table([by_name[k] for k in ['checkpoint_1000','checkpoint_3000','checkpoint_5000','last_64']])}
<details><summary>Reproducibilidad y límites</summary><ul><li>Protocolo, resultados por partida, diferencias emparejadas, metadatos de entrenamiento y modelos experimentales están en <code>{e(str(study.resolve()))}</code>.</li><li>Los intervalos del 99% usan bootstrap por pareja de colores. Son exploratorios: no garantizan cobertura simultánea tras múltiples comparaciones. Solo hay dos semillas de entrenamiento.</li><li>Las continuaciones cambian las trayectorias del autojuego a medida que cambia el modelo, aunque arranquen con la misma semilla.</li><li>No se ha probado entrenar una red mayor desde cero ni se ha demostrado un techo de juego perfecto. La red actual dedica {grouped(audit['parameter_counts']['policy'])} parámetros a la política, frente a {grouped(audit['parameter_counts']['trunk'])} al tronco: contar parámetros totales no mide por sí solo la capacidad estratégica.</li><li>El repaso histórico, las simetrías y los cambios de exploración son variantes experimentales, no resultados del AlphaZero original para este juego.</li></ul></details>
<footer><small>Referencias primarias: <a href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf">AlphaZero (Silver et al.)</a> y <a href="https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf">Policy improvement by planning with Gumbel (Danihelka et al.)</a>. Este último estudia limitaciones de la mejora de política con pocas simulaciones; no demuestra por sí solo la causa de los resultados de Pentomino.</small></footer></html>'''
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(body)
    bundle={'protocol':protocol,'audit':audit,'matches':matches,'comparisons':comparisons,
            'selection':selected,'endgames':endgames,'prior_followup':prior_bundle,'policy_entropy':entropy,
            'symmetry_confirmation':symmetry_bundle,'replay_capacity':capacity_bundle,
            'replay_capacity_confirmation':capacity_confirmation,'search1000':search1000_bundle}
    output.with_suffix('.json').write_text(json.dumps(bundle,indent=2))
    print(output.resolve())


if __name__=='__main__':main()
