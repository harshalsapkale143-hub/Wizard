# Exit Research Next Step

The fine-grid experiment completed successfully but did not improve on the 50-DMA baseline. Next research should focus on structurally different exit logic rather than finer ATR multipliers.

## Planned variants
1. `baseline_50dma`
2. `partial_50dma_then_atr`: take 50% off at +2R, trail remainder with 3x ATR after +2R
3. `partial_50dma_then_20dma`: take 50% off at +2R, trail remainder on 20-DMA close
4. `breakeven_2r_50dma`: move stop to breakeven at +2R, then retain 50-DMA exit
5. `delayed_50dma`: ignore 50-DMA exit until +2R; before +2R use initial stop

All variants should be evaluated on the same four historical windows and configuration grid as the fine-grid experiment, with the same position sizing, costs, and data universe. Report return, profit factor, Sharpe, max drawdown, trade count, win rate, average realized R, and MFE/MAE. Emphasize consistency across windows/configurations and avoid selecting a rule solely because it has the highest aggregate historical return.
