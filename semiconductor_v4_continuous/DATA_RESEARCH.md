# V4 Historical Data Research

Research status as of 2026-08-09. This document concerns the isolated V4
research system. It does not change the 3-day/5-day paper or live bots.

## Recommended Complete Route

The strongest single research stack is an institutional WRDS route:

1. **CRSP US Stock and CRSP/Compustat Merged** for permanent security identity,
   historical names and classifications, active and inactive securities,
   distributions, corporate actions, and delisting outcomes.
2. **Compustat North America Security Daily** or another licensed point-in-time
   source for split-adjusted daily OHLCV, linked through permanent identifiers.
3. **S&P Global Market Intelligence Securities Finance, Equity Americas** for
   daily lendable supply, utilization, availability, and borrowing fees.

Why this route is preferred:

- CRSP's PERMNO follows a security through ticker and name changes and CRSP
  covers active and inactive US securities.
- CRSP/Compustat includes historical GICS intervals rather than only today's
  classification.
- S&P Global's securities-finance product has daily history from 2002 and
  explicitly covers supply, demand, fees, utilization, and availability.
- The products share CUSIP/ISIN-class identifiers, allowing an audited link to
  a stable research security ID.

Official references:

- [CRSP research products and PERMNO](https://www.crsp.org/research/)
- [CRSP/Compustat historical GICS guide](https://wrds-www.wharton.upenn.edu/documents/402/CRSP-Compustat_Merged_Database_Data_Guide_9efDcmD.pdf)
- [Compustat North America on WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/sp-global-market-intelligence/)
- [S&P Global Securities Finance](https://www.spglobal.com/market-intelligence/en/solutions/products/securities-finance)
- [Markit/S&P Securities Finance on WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/markit/)

These products require separate institutional licenses. WRDS access by itself
does not automatically include CRSP, Compustat, or Markit/S&P data.

## Useful But Incomplete Routes

**Massive** provides broad US daily OHLCV flat files and separate split,
dividend, and ticker-event endpoints. Its long-history plan can supply the
price/action side, but it does not provide a historical securities-lending tape
and does not by itself certify historical semiconductor classification.

- [Massive stock flat files](https://massive.com/docs/flat-files/stocks/overview)
- [Massive stock REST and corporate actions](https://massive.com/docs/rest/stocks/overview)

**Sharadar/Nasdaq Data Link** covers active and delisted US companies back to
the 1990s and can be a practical OHLCV/fundamental source. It still needs a
separate historical borrow source, and its classification history must be
verified before setting the point-in-time attestation.

- [Sharadar coverage](https://www.sharadar.com/)
- [Nasdaq Data Link documentation](https://docs.data.nasdaq.com/)

Neither route may be promoted to strict V4 data merely because it contains
many rows. Missing borrow records must be represented explicitly as
unavailable, not silently assigned a cheap fallback rate.

## Frozen Challenge Protocol

The acceptance thresholds and untouched-period boundaries are fixed in
`data_contract.json` before obtaining the new dataset.

- 2007-01-03 through 2021-07-29 is the primary historical challenge period.
- 2021-07-30 through 2026-07-31 is already inspected and is reported
  separately.
- Data after 2026-08-10 remains the forward period.
- V4 receives one unchanged rerun after the bundle passes the data contract.
- Schema, coverage, identity, and action audits are permitted before that run;
  strategy predictions and performance-guided changes are not.

## Canonical Bundle

Run `historical_data.py init` to create a provider-neutral staging bundle. The
bundle uses a permanent research symbol such as `PERMNO:12345`; ticker symbols
are metadata and may change over time.

```text
v4-stage/
  raw/                         exact, unedited vendor exports
  raw_manifest.json            registered file hashes and source metadata
  provenance.json              explicit coverage attestations
  canonical/
    security_master.csv
    stock_bars.csv
    market_bars.csv
    universe_membership.csv
    borrow_costs.csv
    dividends.csv
    corporate_actions.csv
```

The existing V4 CSV schemas remain unchanged. `historical_data.py install`
normalizes a passing bundle, materializes documented delisting recovery tails,
backs up the exploratory files, and installs the strict files only after an
explicit confirmation token.

## Locked Confirmation Sequence

1. Register and verify the immutable raw exports.
2. Normalize them into the canonical tables without editing `raw/`.
3. Run readiness checks only. No V4 prediction is generated at this stage.
4. Install only if all source and sample requirements pass.
5. Freeze the normalized dataset with `confirmation.py freeze-dataset`.
6. Verify both the frozen hypothesis and frozen dataset hashes.
7. Claim and run the one allowed unchanged V4 confirmation with
   `confirmation.py run-once`.
8. Preserve the receipt and all decomposed outputs regardless of whether V4
   passes or fails.

The one-shot report includes full/pre-2021/2021-2026 results, HAC factor alpha,
SOXX and QQQ betas, residual Rank IC and spread, direction calibration, actual
and predicted regime splits, long/short/core/direction contributions, cost and
borrow drag, stress results, every major drawdown, and year/episode/security
concentration.

There are only three valid outcomes:

- Data fails readiness: repair and document the data; do not run V4.
- Data passes and V4 fails: V4 is failed. A redesign is V5 and this history is
  no longer untouched.
- Data passes and V4 passes: freeze the fitted/procedural result and begin a
  prospective paper test; do not jump directly to live deployment.
