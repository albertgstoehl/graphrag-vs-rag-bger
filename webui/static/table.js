/**
 * Shared Alpine.js factory for sortable + filterable tables.
 *
 * Both `/metrics` (per-config aggregates) and `/inspector` (per-query rows)
 * use this. Templates instantiate with `x-data="dataTable(rows, options)"`,
 * declare columns via `<th @click="sort('field')">`, and render rows via
 * `<template x-for="row in visibleRows()">`.
 *
 * Options:
 *   sortKey   default sort column (path; supports nested via "values.x")
 *   sortDir   default direction (1 asc, -1 desc)
 *   filterFields  array of paths to match against the free-text filter
 *                 (defaults to all top-level keys)
 */
function dataTable(rows, options = {}) {
  return {
    rows: rows || [],
    filter: '',
    sortKey: options.sortKey || null,
    sortDir: options.sortDir || -1,
    filterFields: options.filterFields || null,

    sort(key) {
      if (this.sortKey === key) {
        this.sortDir *= -1;
      } else {
        this.sortKey = key;
        this.sortDir = key.startsWith('values.') || /recall|precision|mrr|ndcg|delta|score|rate/.test(key) ? -1 : 1;
      }
    },
    sortIndicator(key) {
      if (this.sortKey !== key) return '';
      return this.sortDir === 1 ? ' ▲' : ' ▼';
    },
    getValue(row, path) {
      return path.split('.').reduce((o, k) => (o == null ? null : o[k]), row);
    },
    visibleRows() {
      const f = this.filter.trim().toLowerCase();
      let arr = this.rows;
      if (f) {
        const fields = this.filterFields || Object.keys(this.rows[0] || {});
        arr = arr.filter(r => fields.some(field => {
          const v = this.getValue(r, field);
          return v != null && String(v).toLowerCase().includes(f);
        }));
      }
      const k = this.sortKey, d = this.sortDir;
      if (!k) return arr;
      return arr.slice().sort((a, b) => {
        let av = this.getValue(a, k);
        let bv = this.getValue(b, k);
        if (av == null) av = -Infinity;
        if (bv == null) bv = -Infinity;
        if (typeof av === 'string' && typeof bv === 'string') {
          return av.localeCompare(bv) * d;
        }
        if (av < bv) return -1 * d;
        if (av > bv) return 1 * d;
        return 0;
      });
    },
    fmt(v, digits = 3) {
      if (v == null) return '–';
      return typeof v === 'number' ? v.toFixed(digits) : v;
    },
    fmtSigned(v, digits = 3) {
      if (v == null) return '–';
      return (v >= 0 ? '+' : '') + v.toFixed(digits);
    },
    /** Cell shading for non-negative metric values. */
    cellShade(v) {
      if (v == null || v === 0) return 'text-slate-400';
      if (v >= 0.5) return 'bg-emerald-100 text-emerald-900';
      if (v >= 0.2) return 'bg-emerald-50';
      return '';
    },
    /** Two-tone shading for delta values (positive green, negative red). */
    deltaShade(v) {
      if (v == null || v === 0) return 'text-slate-400';
      if (v >= 0.5) return 'bg-emerald-200 text-emerald-900';
      if (v >= 0.2) return 'bg-emerald-100';
      if (v <= -0.2) return 'bg-rose-100 text-rose-900';
      return '';
    },
  };
}
