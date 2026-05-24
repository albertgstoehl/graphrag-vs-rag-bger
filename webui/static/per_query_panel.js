/**
 * Alpine factory for the shared per-query drilldown panel that lives
 * embedded under each aggregate metrics view.
 *
 * Usage from a template:
 *   <div x-data="perQueryPanel({homeSystem: 'graph_1hop',
 *                               metricLocked: false,
 *                               urlParam: 'cell'})">
 *
 * Then call `panel.open({mode, system, ranking, k, [metric]})` to open.
 *
 * Options:
 *   homeSystem    default highlighted system (e.g. 'graph_1hop')
 *   metricLocked  if true, the metric selector is hidden (graph-nearness)
 *   urlParam      query string key under which the open state is serialised
 *                 ('cell' for the matrix, 'row' for the flat table,
 *                 'bar' for nearness). Used for deep-linkable reload.
 *   defaultMetric initial metric pick when opening ('recall', 'nearness_score', …)
 *
 * Perf notes (2026-05-10): the table is virtualised — only rows that fall
 * inside the scroll viewport (plus a small overscan) are kept in the DOM.
 * 3000 logical rows → ~30 rendered rows → ~100× fewer Alpine bindings, so
 * sort/filter flips stay snappy. The fetched JSON is also cached per
 * (mode, ranking, k), so re-clicking a cell is instant.
 */

// Module-level cache shared across all panel instances on the page.
// Keys: 'pool' | 'ranked|<ranking>|<k>'. Values: {rows, systems}.
const _PQ_CACHE = new Map();

function perQueryPanel(opts = {}) {
  return {
    // ------------------- state -------------------
    open: false,
    loading: false,
    mode: 'ranked',          // 'ranked' | 'pool'
    system: opts.homeSystem || 'graph_1hop',
    ranking: 'cross_encoder',
    k: 10,
    language: 'all',         // 'all' | 'de' | 'fr' | 'it'
    metric: opts.defaultMetric || 'recall',
    metricLocked: !!opts.metricLocked,
    rows: [],                // raw fetched rows (with computed `delta`)
    systems: [],
    sortKey: 'delta',
    sortDir: -1,
    filter: '',              // bound to the search input
    filterApplied: '',       // debounced filter actually used in the slice
    _filterTimer: null,
    urlParam: opts.urlParam || 'cell',

    // ----- virtualisation state -----
    rowHeight: 36,           // matches td px-3 py-2 (~36 px). Recalc after first render.
    scrollTop: 0,
    viewportHeight: 480,     // arbitrary default; updated on layout
    overscan: 8,
    _processed: [],          // memoised filter+sort output for the current rows
    _processedKey: '',       // key to invalidate _processed cache

    /** Called once from the outer x-data's x-init. Alpine's `init()`
        only auto-runs on top-level scope objects, and this panel lives
        nested inside an outer scope, so we expose setup() and the host
        template invokes it explicitly. */
    setup() {
      const sp = new URLSearchParams(window.location.search);
      const v = sp.get(this.urlParam);
      if (v) this.openFromString(v);
    },

    // ------------------- public open/close -------------------
    /** Open the panel for a (mode, system, ranking, k, language) target. */
    openFor({mode, system, ranking, k, metric, language}) {
      this.mode = mode || 'ranked';
      if (system) this.system = system;
      if (ranking) this.ranking = ranking;
      if (k) this.k = Number(k);
      if (metric) this.metric = metric;
      if (language) this.language = language;
      this.open = true;
      this.persistUrl();
      this.refetch();
    },

    /** Parse a serialised cell key, e.g. 'graph_1hop|cross_encoder_10' or
        'graph_1hop|post_cap'. Used for both deep-link rehydration and direct
        cell-click on the matrix. Optional `metric` arg threads the matrix's
        current metric ('recall' or 'precision') into the drilldown so the
        panel doesn't reset to recall every time the user clicks. */
    openFromString(s, metric) {
      const [system, stage] = (s || '').split('|');
      if (!system || !stage) return;
      const m = metric || 'recall';
      if (stage === 'post_cap') {
        this.openFor({mode: 'pool', system, metric: m});
        return;
      }
      const rk = stage.match(/^(cosine|cross_encoder|indegree)_(\d+)$/);
      if (rk) {
        this.openFor({mode: 'ranked', system, ranking: rk[1], k: Number(rk[2]),
                      metric: m});
        return;
      }
      // pool stage other than post_cap: aggregate-only, no per-query data.
      this.open = false;
    },

    /** Open from (system, ranking, k) directly — used by /metrics flat table
        and /metrics/graph-nearness bar click. */
    openRanked({system, ranking, k, metric, language}) {
      this.openFor({mode: 'ranked', system, ranking, k, metric, language});
    },
    setLanguage(l) { this.language = l; if (this.open) this.refetch(); },

    close() {
      this.open = false;
      this.persistUrl();
    },

    // ------------------- URL deep-link state -------------------
    persistUrl() {
      const sp = new URLSearchParams(window.location.search);
      if (!this.open) {
        sp.delete(this.urlParam);
      } else if (this.mode === 'pool') {
        sp.set(this.urlParam, `${this.system}|post_cap`);
      } else {
        sp.set(this.urlParam, `${this.system}|${this.ranking}_${this.k}`);
      }
      const qs = sp.toString();
      const url = window.location.pathname + (qs ? '?' + qs : '');
      window.history.replaceState(null, '', url);
    },

    setRanking(r) { this.ranking = r; this.persistUrl(); if (this.open) this.refetch(); },
    setK(k)       { this.k = k;       this.persistUrl(); if (this.open) this.refetch(); },
    setMetric(m)  { this.metric = m;  if (this.open) this.rerunDelta(); },

    // ------------------- data fetch -------------------
    _cacheKey() {
      const lang = this.language || 'all';
      return this.mode === 'pool'
        ? `pool|${lang}`
        : `ranked|${this.ranking}|${this.k}|${lang}`;
    },

    async refetch() {
      const key = this._cacheKey();
      const cached = _PQ_CACHE.get(key);
      if (cached) {
        // Hit: skip the network round-trip entirely.
        this.systems = cached.systems;
        this.rows = cached.rows.map(row => {
          row.delta = this.computeDelta(row);
          return row;
        });
        this.loading = false;
        this._invalidateProcessed();
        this._resetScroll();
        return;
      }
      this.loading = true;
      this.rows = [];
      const lang = this.language || 'all';
      let url;
      if (this.mode === 'pool') {
        url = `/api/per-query?mode=pool&language=${lang}`;
      } else {
        url = `/api/per-query?mode=ranked&ranking=${this.ranking}&k=${this.k}&language=${lang}`;
      }
      try {
        const r = await fetch(url, {headers: {'Accept': 'application/json'}});
        if (!r.ok) {
          this.rows = [];
          this.systems = [];
        } else {
          const j = await r.json();
          this.systems = j.systems || [];
          const rows = (j.rows || []).map(row => {
            row.delta = this.computeDelta(row);
            return row;
          });
          this.rows = rows;
          // Cache only non-empty results; empty means "Stage 3 not done yet".
          if (rows.length) {
            _PQ_CACHE.set(key, {rows: rows.slice(), systems: j.systems || []});
          }
        }
      } catch (e) {
        this.rows = [];
      } finally {
        this.loading = false;
        this._invalidateProcessed();
        this._resetScroll();
      }
    },

    // ------------------- helpers -------------------
    /** Pool-mode metrics are recall + precision; ranked-mode adds
        mrr/ndcg/nearness_score. When the panel switches modes, snap the
        metric to a value the new mode can resolve. */
    metricForMode() {
      if (this.mode === 'pool' && this.metric !== 'recall' && this.metric !== 'precision') {
        return 'recall';
      }
      return this.metric;
    },
    cellValue(row, system) {
      const cell = row.values?.[system];
      if (cell == null) return null;
      // Both modes now ship nested `{recall, precision[, ...]}` per system.
      return cell?.[this.metricForMode()];
    },
    computeDelta(row) {
      const m = this.metricForMode();
      const a = row.values?.graph_1hop?.[m];
      const b = row.values?.emb_1hop?.[m];
      if (a == null || b == null) return null;
      return a - b;
    },
    /** Recompute delta when the metric flips (ranked or pool). Single
        linear pass over the existing array; no IO. */
    rerunDelta() {
      for (let i = 0; i < this.rows.length; i++) {
        this.rows[i].delta = this.computeDelta(this.rows[i]);
      }
      this._invalidateProcessed();
    },

    sort(key) {
      if (this.sortKey === key) {
        this.sortDir *= -1;
      } else {
        this.sortKey = key;
        this.sortDir = -1;
      }
      this._invalidateProcessed();
      this._resetScroll();
    },
    sortIndicator(key) {
      if (this.sortKey !== key) return '';
      return this.sortDir === 1 ? ' ▲' : ' ▼';
    },
    /** Resolve the sort key to a numeric value on a row. Supports
        'delta', 'language', 'year', 'n_gt', 'query_id', or 'row.<system>'
        which means "value of this row's metric for that system". */
    sortValue(row, key) {
      if (key === 'delta')      return row.delta;
      if (key === 'language')   return row.language;
      if (key === 'year')       return row.year;
      if (key === 'n_gt')       return row.n_gt;
      if (key === 'query_id')   return row.query_id;
      if (key.startsWith('row.')) {
        const s = key.slice(4);
        return this.cellValue(row, s);
      }
      return null;
    },

    // ----- filter input (debounced) -----
    onFilterInput() {
      // Debounce so each keystroke doesn't re-sort 3000 rows.
      if (this._filterTimer) clearTimeout(this._filterTimer);
      this._filterTimer = setTimeout(() => {
        this.filterApplied = this.filter.trim().toLowerCase();
        this._invalidateProcessed();
        this._resetScroll();
      }, 200);
    },

    /** Memoised filter+sort. Recomputed only when sort/filter/metric/rows
        change. Returns the FULL ordered, filtered array — virtualisation
        further slices it before render. */
    processedRows() {
      const want = `${this.sortKey}|${this.sortDir}|${this.filterApplied}|${this.metric}|${this.rows.length}`;
      if (want === this._processedKey && this._processed.length) {
        return this._processed;
      }
      const f = this.filterApplied;
      let arr = this.rows;
      if (f) {
        arr = arr.filter(r =>
          (r.query_id && r.query_id.toLowerCase().includes(f)) ||
          (r.language && r.language.toLowerCase().includes(f)) ||
          (r.year != null && String(r.year).includes(f))
        );
      }
      const k = this.sortKey, d = this.sortDir;
      const sorted = arr.slice().sort((a, b) => {
        let av = this.sortValue(a, k);
        let bv = this.sortValue(b, k);
        if (av == null) av = -Infinity;
        if (bv == null) bv = -Infinity;
        if (typeof av === 'string' && typeof bv === 'string') {
          return av.localeCompare(bv) * d;
        }
        if (av < bv) return -1 * d;
        if (av > bv) return 1 * d;
        return 0;
      });
      this._processed = sorted;
      this._processedKey = want;
      return sorted;
    },
    _invalidateProcessed() {
      this._processedKey = '';
      this._processed = [];
    },

    // ----- virtualisation -----
    /** Bind to the scroll container's @scroll. Updates scrollTop from the
        live element. */
    onScroll(ev) {
      this.scrollTop = ev.target.scrollTop;
    },
    /** Called from x-init on the scroll container so we can size the
        viewport once the layout has settled. */
    initScroller(el) {
      this._scrollEl = el;
      // Reasonable default: cap at ~70 % viewport, min 360 px.
      const h = Math.max(360, Math.min(window.innerHeight * 0.7, 720));
      this.viewportHeight = h;
      el.style.maxHeight = h + 'px';
      // Re-measure row height after a data row paints. Skip the aria-hidden
      // spacer rows whose height is just the padTop/padBottom CSS value;
      // measuring those would collapse the virtual list height.
      const measure = () => {
        const trs = el.querySelectorAll('tbody tr:not([aria-hidden])');
        for (const tr of trs) {
          if (tr.offsetHeight >= 20) {
            this.rowHeight = tr.offsetHeight;
            return true;
          }
        }
        return false;
      };
      // Try a few frames; the table is conditionally rendered via x-if so
      // it may not be in the DOM on the first paint.
      let tries = 0;
      const tick = () => {
        if (measure()) return;
        if (++tries < 10) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    },
    _resetScroll() {
      this.scrollTop = 0;
      if (this._scrollEl) this._scrollEl.scrollTop = 0;
    },
    /** Total height of the (virtual) list — drives the scrollbar via a
        spacer div. */
    totalHeight() {
      return this.processedRows().length * this.rowHeight;
    },
    /** Index range currently in-DOM. */
    visibleRange() {
      const all = this.processedRows();
      const total = all.length;
      if (!total) return {start: 0, end: 0, padTop: 0, padBottom: 0};
      const rh = this.rowHeight || 36;
      const visCount = Math.ceil(this.viewportHeight / rh) + this.overscan * 2;
      let start = Math.floor(this.scrollTop / rh) - this.overscan;
      if (start < 0) start = 0;
      let end = start + visCount;
      if (end > total) end = total;
      return {
        start, end,
        padTop: start * rh,
        padBottom: (total - end) * rh,
      };
    },
    /** Subset of rows that should actually render in the DOM. */
    visibleRows() {
      const {start, end} = this.visibleRange();
      return this.processedRows().slice(start, end);
    },

    fmt(v, digits = 3) {
      if (v == null) return '–';
      return typeof v === 'number' ? v.toFixed(digits) : v;
    },
    fmtSigned(v, digits = 3) {
      if (v == null) return '–';
      return (v >= 0 ? '+' : '') + v.toFixed(digits);
    },
    cellShade(v) {
      if (v == null || v === 0) return 'text-slate-400';
      if (v >= 0.5) return 'bg-emerald-100 text-emerald-900';
      if (v >= 0.2) return 'bg-emerald-50';
      return '';
    },
    deltaShade(v) {
      if (v == null || v === 0) return 'text-slate-400';
      if (v >= 0.2) return 'bg-emerald-200 text-emerald-900';
      if (v >= 0.05) return 'bg-emerald-100';
      if (v <= -0.2) return 'bg-rose-100 text-rose-900';
      return '';
    },
    title() {
      if (!this.open) return '';
      if (this.mode === 'pool') return `${this.system} · post_cap pool`;
      return `${this.system} · ${this.ranking} · k=${this.k}`;
    },
    statusLine() {
      if (this.loading) return '';
      if (!this.rows.length) return '';
      const total = this.rows.length;
      const shown = this.processedRows().length;
      const tail = shown === total ? '' : ` (filtered from ${total})`;
      return `${shown} queries${tail} · sort: ${this.sortKey} ${this.sortDir > 0 ? '▲' : '▼'}`;
    },
    openDetail(qid) {
      const view = this.mode === 'pool' ? 'pool' : 'ranked';
      const left = this.system;
      const right = this.system === 'emb_1hop' ? 'graph_1hop' : 'emb_1hop';
      window.location = `/inspector/${qid}?view=${view}&left=${left}&right=${right}`;
    },
  };
}
