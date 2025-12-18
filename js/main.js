// ===== Vision reveal (stable hysteresis + play animation once per enter) =====
(() => {
  const items = document.querySelectorAll('.vision-shot.reveal');
  if (!items.length) return;

  // Clean state on load (avoid bfcache weirdness)
  items.forEach(el => {
    el.classList.remove('is-visible', 'play-in');
  });

  // Hysteresis: enter high, exit very low (prevents boundary spam)
  const ENTER_AT = 0.32;
  const EXIT_AT  = 0.03;

  // A few thresholds helps smoother updates, but logic uses ENTER/EXIT
  const thresholds = [0, 0.03, 0.08, 0.15, 0.25, 0.32, 0.5];

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const el = e.target;
      const r = e.intersectionRatio;

      const visible = el.classList.contains('is-visible');

      // ENTER: only if not already visible
      if (!visible && r >= ENTER_AT) {
        el.classList.add('is-visible');

        // trigger fly-in animation once
        el.classList.add('play-in');
      }

      // EXIT: only if currently visible and really out
      if (visible && r <= EXIT_AT) {
        el.classList.remove('is-visible');
        // re-arm: next enter can animate again
      }
    }
  }, {
    root: null,
    threshold: thresholds,
    rootMargin: '0px 0px -8% 0px' // slightly delays exit while you “read”
  });

  items.forEach(el => {
    // remove play-in after animation completes so it won't keep replaying
    el.addEventListener('animationend', () => {
      el.classList.remove('play-in');
    });
    io.observe(el);
  });
})();
