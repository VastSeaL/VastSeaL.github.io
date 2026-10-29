/* Run before CSS to avoid a flash when restoring a saved theme. */
(function () {
  let theme;
  try { theme = localStorage.getItem('academic-theme'); } catch (_) { /* Storage is optional. */ }
  if (theme !== 'dark' && theme !== 'light') theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
})();
