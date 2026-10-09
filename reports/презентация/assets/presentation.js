(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const el = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined) node.textContent = text; return node; };
  const names = {pooled_seasonal:'Общая сезонность',causal_selector:'Выбор по прошлым ошибкам',prophet:'Prophet',ensemble:'Ансамбль',catboost:'CatBoost',lightgbm:'LightGBM',chronos:'Chronos-Bolt',naive_last:'Последнее значение',naive_seasonal:'Сезонный наивный',short_history:'Короткая история'};
  const title = $('body > h1');
  if (!title) return;
  const subtitle = title.nextElementSibling;
  const sections = [...document.querySelectorAll('body > section')];
  const top = el('header','topbar');
  const brand = el('a','brand'); brand.href='#overview'; brand.append(el('span',null,'↗'),document.createTextNode('СберИндекс / исследование'));
  const nav = el('nav'); nav.setAttribute('aria-label','Разделы презентации');
  [['methodology','Методология'],['forecasts','Прогнозы'],['detectors','Изменения'],['limitations','Ограничения']].forEach(([id,label])=>{const a=el('a',null,label);a.href='#'+id;nav.append(a)});
  const report=el('a','report-link','Читать отчёт ↗');report.href='../методологический_отчёт.md';top.append(brand,nav,report);document.body.prepend(top);
  const main=el('main');document.body.insertBefore(main,title);
  const hero=el('div','hero');hero.id='overview';hero.append(el('div','eyebrow','Аналитика муниципального потребления'));
  title.replaceChildren(document.createTextNode('Потребление.'),el('br'),el('span','hero-accent','Увидеть изменения.'),el('br'),document.createTextNode('Построить прогноз.'));hero.append(title,subtitle);
  const bottom=el('div','hero-bottom');const cta=el('a',null,'Посмотреть результаты ↓');cta.href='#forecasts';bottom.append(cta,el('span','pill','История · 2023–2024'),el('span','pill','Временная валидация'));hero.append(bottom);main.append(hero);
  const stats=el('div','stats');
  const originalText=subtitle.textContent;
  const territories=originalText.match(/([\d\s]+)\s*МО/);
  const categories=originalText.match(/(\d+)\s*категор/);
  const months=originalText.match(/(\d+)\s*месяц/);
  [[territories?territories[1].trim():'—','муниципальных образований'],[categories?categories[1]:'—','категорий расходов'],[months?months[1]:'—','месяца наблюдений'],['1 / 3 / 6 / 12','горизонты, месяцев']].forEach(([v,label])=>{const card=el('div','stat');card.append(el('strong',null,v),el('span',null,label));stats.append(card)});main.append(stats);
  const ids=['methodology','forecasts','examples','detectors'];
  sections.forEach((section,i)=>{
    if(!section.id)section.id=ids[i]||'evidence-'+i;
    const heading=$('h2',section);heading.dataset.number=String(i+1).padStart(2,'0');
    if(heading.textContent==='Ограничения'){section.id='limitations';section.classList.add('limits')}
    if(heading.textContent==='Источники и лицензии'){section.id='sources';section.classList.add('sources')}
    main.append(section);
    section.querySelectorAll('img').forEach(img=>{img.loading='lazy';if(!img.alt)img.alt=heading.textContent});
    section.querySelectorAll('table').forEach(table=>{table.removeAttribute('border');const wrapper=el('div','table-scroll');wrapper.tabIndex=0;wrapper.setAttribute('role','region');wrapper.setAttribute('aria-label',heading.textContent+' — таблица');table.before(wrapper);wrapper.append(table)});
    if(i>=4&&!section.classList.contains('limits')&&!section.classList.contains('sources')){
      section.classList.add('evidence');const table=$('.table-scroll',section);if(table){const details=el('details');details.append(el('summary',null,'Открыть подробные результаты'));table.before(details);details.append(table)}
    }
  });
  const forecasting=$('#forecasts'),table=$('table',forecasting),select=$('#horizon');
  if(table&&select){
    const rows=[...table.tBodies[0].rows];
    const data=rows.map(row=>({row,model:row.cells[0].textContent.trim(),horizon:row.cells[1].textContent.trim(),mae:Number(row.cells[3].textContent.trim()),pairs:row.cells[2].textContent.trim()}));
    const milestones=el('div','milestones');
    ['1','3','6','12'].forEach(h=>{
      const subset=data.filter(d=>d.horizon===h&&Number.isFinite(d.mae)).sort((a,b)=>a.mae-b.mae);
      if(!subset.length)return;
      const best=subset[0];const card=el('a','milestone');card.href='#forecasts';
      card.append(el('span','milestone-period',h+' '+(h==='1'?'месяц':h==='3'?'месяца':'месяцев')),el('strong',null,best.mae.toLocaleString('ru-RU',{maximumFractionDigits:2})+' ₽'),el('span','milestone-model',names[best.model]||best.model),el('span','milestone-caption','Минимальная MAE на общей выборке'));
      card.addEventListener('click',()=>{select.value=h;select.dispatchEvent(new Event('change'))});milestones.append(card);
    });
    const resultIntro=el('div','result-intro');resultIntro.append(el('span','kicker','Результаты в четырёх горизонтах'),el('h2',null,'Один подход не выигрывает везде.'),el('p',null,'Короткую историю сравниваем на одинаковых фактах. Для каждого горизонта — собственный результат и границы его интерпретации.'));
    stats.after(resultIntro,milestones);
    const visual=el('div','hero-visual');visual.append(el('div','visual-caption','MAE на одинаковых наблюдениях'));
    const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 400 250');svg.setAttribute('role','img');svg.setAttribute('aria-label','MAE общей сезонности и Prophet на четырёх горизонтах');
    const svgEl=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e};
    const chartModels=['pooled_seasonal','prophet'];const horizons=['1','3','6','12'];const values=data.filter(d=>chartModels.includes(d.model)&&Number.isFinite(d.mae));const upper=Math.max(...values.map(d=>d.mae),1)*1.12;
    [0,.5,1].forEach(t=>{const y=200-t*170;svgEl('line',{x1:44,y1:y,x2:380,y2:y,stroke:'#ffffff16'});svgEl('text',{x:34,y:y+4,fill:'#96b6af','font-size':10,'text-anchor':'end'},Math.round(t*upper).toLocaleString('ru-RU'))});
    horizons.forEach((h,i)=>svgEl('text',{x:50+i*106,y:227,fill:'#afc7bf','font-size':11,'text-anchor':'middle'},h+' мес.'));
    chartModels.forEach((model,j)=>{const points=horizons.map((h,i)=>{const d=data.find(d=>d.model===model&&d.horizon===h);return d&&Number.isFinite(d.mae)?[50+i*106,200-d.mae/upper*170]:null});const color=j?'#8cafa9':'#d3f580';let path='';let open=false;points.forEach(p=>{if(!p){open=false;return}path+=(open?' L':' M')+p.join(',');open=true});svgEl('path',{d:path,fill:'none',stroke:color,'stroke-width':3,'stroke-linecap':'round','stroke-linejoin':'round'});points.filter(Boolean).forEach(([x,y])=>svgEl('circle',{cx:x,cy:y,r:4,fill:color,stroke:'#183c38','stroke-width':3}))});visual.append(svg);
    const legend=el('div','visual-legend');legend.append(el('span',null,'Общая сезонность'),el('span',null,'Prophet'));visual.append(legend,el('p','visual-footnote','Историческая оценка · ₽ · ниже лучше'));hero.append(visual);
    const tabs=el('div','horizon-tabs');tabs.setAttribute('role','group');tabs.setAttribute('aria-label','Горизонт прогнозирования');
    [['1','1 месяц'],['3','3 месяца'],['6','6 месяцев'],['12','12 месяцев'],['all','Все']].forEach(([value,label])=>{const button=el('button',null,label);button.type='button';button.dataset.horizon=value;button.addEventListener('click',()=>{select.value=value;select.dispatchEvent(new Event('change'))});tabs.append(button)});select.closest('label').before(tabs);select.closest('label').classList.add('native-horizon');
    const chart=el('div','chart');chart.setAttribute('role','region');chart.setAttribute('aria-label','Сравнение MAE моделей');chart.setAttribute('aria-live','polite');
    $('.table-scroll',forecasting).before(chart);
    function render(){
      const horizon=select.value;chart.replaceChildren();
      tabs.querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.horizon===horizon)));
      const selected=data.filter(d=>horizon==='all'||d.horizon===horizon);
      const plotted=selected.filter(d=>d.horizon===(horizon==='all'?'1':horizon)&&Number.isFinite(d.mae)).sort((a,b)=>a.mae-b.mae);
      rows.forEach(row=>row.classList.remove('winner'));
      if(!plotted.length)return;
      plotted[0].row.classList.add('winner');
      const chartTitle=el('div','chart-title');chartTitle.append(el('strong',null,'Ошибка прогноза · MAE, ₽'),el('span',null,'Меньше — лучше'));chart.append(chartTitle);
      const max=Math.max(...plotted.map(d=>d.mae),1);
      plotted.forEach((d,i)=>{const line=el('div','bar-row'+(i===0?' best':'')+(d.model==='prophet'?' baseline':''));const label=el('span','bar-label',names[d.model]||d.model);label.title=d.model;const track=el('div','bar-track');const bar=el('div','bar-fill');bar.style.width=(d.mae/max*100)+'%';track.append(bar);line.append(label,track,el('span','bar-value',d.mae.toLocaleString('ru-RU',{maximumFractionDigits:2})));chart.append(line)});
      chart.append(el('p','chart-note',(horizon==='all'?'Диаграмма: горизонт 1 месяц. ':'')+plotted[0].pairs+' совпадающих фактов. Зелёный — минимальная MAE, светло-зелёный — Prophet. Рейтинг описывает эту выборку; устойчивость преимущества проверяется отдельно.'));
    }
    select.addEventListener('change',render);
    select.value='1';select.dispatchEvent(new Event('change'));
    const plot=$('img',forecasting);if(plot){const details=el('details');details.append(el('summary',null,'Сводный график всех горизонтов · MAE и R²'));plot.before(details);details.append(plot)}
  }
  const method=$('#methodology');
  if(method){
    const flow=el('div','method-flow');
    [['01','Данные','Расходы, доступность рынков и транспортные связи.'],['02','Признаки','История до среза, календарь, макро и соседи.'],['03','Прогноз','Модели, сравнение MAE и выбор по прошлым ошибкам.'],['04','Изменения','Отклонения от прогноза и сигналы соседних МО.']].forEach(([number,label,desc])=>{const tile=el('div','flow-step');tile.append(el('span','flow-number',number),el('h3',null,label),el('p',null,desc));flow.append(tile)});
    $('h2',method).after(flow);
    const image=$('img',method);if(image){const details=el('details');details.append(el('summary',null,'Полная архитектура решения'));image.before(details);details.append(image)}
  }
  const examples=$('#examples');
  if(examples){const exampleTable=$('table',examples);if(exampleTable){const cards=el('div','example-cards');[...exampleTable.tBodies[0].rows].forEach((row,i)=>{if(row.cells.length<7)return;const c=[...row.cells].map(cell=>cell.textContent.trim());const card=el('article','example-card');card.append(el('span','example-tag',['Минимальная ошибка','Медианная ошибка','Максимальная ошибка'][i]||'Наблюдение'),el('h3',null,c[1]),el('p','example-place','МО '+c[0]+' · '+c[3]));const values=el('div','example-values');[['Факт',c[4]],['Прогноз',c[5]],['Абсолютная ошибка',c[6]]].forEach(([label,v])=>{const item=el('div');item.append(el('span',null,label),el('strong',null,Number(v).toLocaleString('ru-RU',{maximumFractionDigits:2})+' ₽'));values.append(item)});card.append(values);cards.append(card)});const wrapper=$('.table-scroll',examples);wrapper.before(cards);const details=el('details');details.append(el('summary',null,'Исходные наблюдения и даты среза'));wrapper.before(details);details.append(wrapper)}}
  const evidence=[...main.querySelectorAll('section.evidence')];if(evidence.length){const area=el('div','evidence-area');const header=el('div','evidence-heading');header.append(el('span','kicker','Проверка результатов'),el('h2',null,'Что стоит за цифрами'),el('p',null,'Покрытие, интервалы неопределённости и независимые эксперименты. Подробные таблицы раскрываются по нажатию.'));const grid=el('div','evidence-grid');evidence[0].before(area);area.append(header,grid);evidence.forEach(section=>grid.append(section))}
  const labels={model:'Модель',horizon:'Горизонт',n_pairs:'Фактов',mae:'MAE, ₽',rmse:'RMSE, ₽',mape:'MAPE, %',r2:'R²',detector:'Детектор',precision:'Точность',recall:'Полнота',micro_f1:'Micro-F1',micro_precision:'Micro-точность',micro_recall:'Micro-полнота',territory_id:'МО',category:'Категория',cutoff:'Начало прогноза',date:'Месяц',actual:'Факт, ₽',predicted:'Прогноз, ₽',absolute_error:'Ошибка, ₽'};
  document.querySelectorAll('th').forEach(th=>{const value=th.textContent.trim();if(labels[value]){th.title=value;th.textContent=labels[value]}});
  document.querySelectorAll('td:first-child').forEach(td=>{const value=td.textContent.trim();if(names[value]){td.title=value;td.textContent=names[value]}});
  const footer=el('footer','footer');footer.append(el('span',null,'СберИндекс · Прогнозирование и структурные изменения'),el('span',null,'Данные → модели → проверяемые выводы'));main.append(footer);
  if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){nav.querySelectorAll('a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id))}})},{rootMargin:'-15% 0px -65% 0px'});['methodology','forecasts','detectors','limitations'].forEach(id=>{const node=document.getElementById(id);if(node)observer.observe(node)})}
})();
