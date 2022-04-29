# 100 shares per trade
HMAX_NORMALIZE = 100
# account balance
INITIAL_ACCOUNT_BALANCE=1000000
# number of stocks
STOCK_DIM = 30
# transaction fee: 0.1%
TRANSACTION_FEE_PERCENT = 0.1 * (1 / 100)
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4
#  Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
OBS_SHAPE_DIM = 181