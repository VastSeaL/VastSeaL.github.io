'use strict';

(() => {
  const root = document.documentElement;
  const toggle = document.querySelector('#theme-toggle');
  const syncTheme = () => {
    const dark = root.dataset.theme === 'dark';
    toggle.setAttribute('aria-pressed', String(dark));
    toggle.setAttribute('aria-label', dark ? '切换浅色模式' : '切换深色模式');
    toggle.title = dark ? '切换浅色模式' : '切换深色模式';
  };
  let explicitTheme = false;
  try { explicitTheme = ['dark', 'light'].includes(localStorage.getItem('academic-theme')); } catch (_) {}
  syncTheme();
  toggle.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    explicitTheme = true;
    try { localStorage.setItem('academic-theme', root.dataset.theme); } catch (_) {}
    syncTheme();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
    if (!explicitTheme) { root.dataset.theme = event.matches ? 'dark' : 'light'; syncTheme(); }
  });

  document.querySelectorAll('dialog').forEach(dialog => {
    dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      const bounds = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
    });
  });

  const search = document.querySelector('#search-dialog');
  const input = document.querySelector('#search-input');
  const results = document.querySelector('#search-results');
  const status = document.querySelector('#search-status');
  let indexPromise;
  let searchVersion = 0;
  const loadIndex = () => indexPromise || (indexPromise = fetch(search.dataset.index)
    .then(response => { if (!response.ok) throw new Error('Search index unavailable'); return response.json(); })
    .catch(error => { indexPromise = null; throw error; }));
  const highlight = (element, text, terms) => {
    const pattern = new RegExp('(' + terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
    text.split(pattern).forEach((part, i) => {
      if (i % 2) { const mark = document.createElement('mark'); mark.textContent = part; element.append(mark); }
      else element.append(document.createTextNode(part));
    });
  };
  const runSearch = async () => {
    const version = ++searchVersion;
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    results.replaceChildren();
    if (!terms.length) { status.textContent = '输入关键词，查找所有文章。'; return; }
    status.textContent = '正在查找…';
    try {
      const index = await loadIndex();
      if (version !== searchVersion) return;
      const matches = index.filter(post => terms.every(term => (post.title + ' ' + post.content).toLowerCase().includes(term)));
      status.textContent = matches.length ? `找到 ${matches.length} 篇文章` : '没有找到相关文章，试试其他关键词。';
      for (const post of matches) {
        const a = document.createElement('a');
        a.href = post.url;
        const title = document.createElement('h3');
        highlight(title, post.title, terms);
        const date = document.createElement('time');
        date.textContent = post.date;
        const excerpt = document.createElement('p');
        const start = Math.max(0, post.content.toLowerCase().indexOf(terms[0]) - 35);
        highlight(excerpt, (start ? '…' : '') + post.content.slice(start, start + 160) + '…', terms);
        a.append(title, date, excerpt);
        results.append(a);
      }
    } catch (_) {
      if (version === searchVersion) status.textContent = '暂时无法加载搜索，请检查连接后重新输入。';
    }
  };
  const openSearch = () => { if (!search.open) search.showModal(); input.focus(); runSearch(); };
  document.querySelectorAll('[data-open-search]').forEach(button => button.addEventListener('click', openSearch));
  input.addEventListener('input', runSearch);
  document.addEventListener('keydown', event => {
    const editing = event.target instanceof Element && (event.target.matches('input, textarea, select') || event.target.isContentEditable);
    if (event.key === '/' && !editing && !document.querySelector('dialog[open]') && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); openSearch(); }
  });

  const copyText = async text => {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    throw new Error('Clipboard unavailable');
  };
  document.querySelectorAll('figure.highlight, .paper-details pre').forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-code'; button.type = 'button'; button.textContent = '复制'; button.setAttribute('aria-label', '复制代码');
    block.style.position = 'relative';
    button.addEventListener('click', async () => {
      const code = block.querySelector('td.code pre, code') || block.querySelector('pre');
      try { await copyText(code.innerText); button.textContent = '已复制'; }
      catch (_) { button.textContent = '请选中复制'; }
      setTimeout(() => { button.textContent = '复制'; }, 1800);
    });
    block.append(button);
  });

  const imageDialog = document.querySelector('#image-dialog');
  let imageOpener;
  document.querySelectorAll('.article-content img').forEach(img => {
    if (img.closest('a')) return;
    img.tabIndex = 0; img.setAttribute('role', 'button'); img.setAttribute('aria-label', '放大图片：' + (img.alt || '文章配图'));
    const open = () => {
      const preview = imageDialog.querySelector('img');
      preview.src = img.currentSrc || img.src; preview.alt = img.alt;
      imageDialog.querySelector('p').textContent = img.alt;
      imageOpener = img; imageDialog.showModal();
    };
    img.addEventListener('click', open);
    img.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
  });
  imageDialog.addEventListener('close', () => { if (imageOpener) imageOpener.focus({ preventScroll: true }); });
  const contents = document.querySelector('.toc-panel');
  if (contents && matchMedia('(max-width: 700px)').matches) contents.open = false;
})();
