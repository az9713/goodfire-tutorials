document.addEventListener('DOMContentLoaded',()=>{
  for(const pre of document.querySelectorAll('pre')){
    if(pre.closest('.codewrap'))continue;
    const wrap=document.createElement('div');wrap.className='codewrap';
    pre.parentNode.insertBefore(wrap,pre);wrap.appendChild(pre);
    const button=document.createElement('button');button.className='copybtn';button.type='button';button.textContent='Copy';
    button.addEventListener('click',async()=>{
      const text=(pre.querySelector('code')||pre).textContent;
      try{await navigator.clipboard.writeText(text);}catch(e){
        const field=document.createElement('textarea');field.value=text;field.style.position='fixed';field.style.opacity='0';document.body.appendChild(field);field.select();const ok=document.execCommand('copy');field.remove();if(!ok){button.textContent='Select text to copy';return;}
      }
      button.textContent='Copied';setTimeout(()=>button.textContent='Copy',1500);
    });wrap.appendChild(button);
  }
  const search=document.getElementById('source-search');
  if(search)search.addEventListener('input',()=>{const q=search.value.toLowerCase();for(const node of document.querySelectorAll('[data-search]'))node.hidden=!node.dataset.search.toLowerCase().includes(q);});
  for(const panel of document.querySelectorAll('[data-demo="relu"]')){
    const slider=panel.querySelector('input[type=range]');const select=panel.querySelector('select');const output=panel.querySelector('output');
    const update=()=>{const x=Number(slider.value);let plus=Math.max(x,0),minus=Math.max(-x,0);if(select.value==='plus')plus=0;if(select.value==='minus')minus=0;output.textContent=`x = ${x.toFixed(1)}; h+ = ${plus.toFixed(1)}; h− = ${minus.toFixed(1)}; y = ${(plus+minus).toFixed(1)}`;};
    slider.addEventListener('input',update);select.addEventListener('change',update);update();
  }
});
