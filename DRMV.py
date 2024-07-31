import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import quantstats as qs
import pandas_market_calendars as mcal
import gurobipy as gp

stocks = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV']
#stocks = ["SPY","QQQ","VNQ","IEF","TLT","TIP","DBC","GLD"]
#stocks = ["SPY", "APA", "INFI", "JPM", "NVAX"]
#stocks = ['USO','GLD','DBB','DBA','GSG','VNQ','DBO','UNG','CORN','SOYB','SLV','PPLT','IYR']
#stocks = ['USO','GLD','DBB','GSG','VNQ','UNG','CORN','SLV', 'VGLT']
#stocks = ['SPY','AGG','DBC','GLD','VNQ']

#df1
start = '2023-03-01'
#end = '2023-10-01'
data = pd.DataFrame()

# Fetch the data for each stock and concatenate it to the `data` DataFrame
for stock in stocks:
    raw = yf.download(stock, start=start)
    raw['Symbol'] = stock  # Add a column indicating the stock symbol
    data = pd.concat([data, raw], axis=0)
df = portfolio_data = data.pivot_table(index='Date', columns='Symbol', values='Adj Close')

#df2
# df = yf.download(stocks)['Adj Close'].dropna()

#Today
placeholder_date = pd.Timestamp.today()
# Assign the values from the last row to the placeholder date
df.loc[placeholder_date] = df.iloc[-1].values


df_returns = df.pct_change().fillna(0)
(1+df_returns).cumprod().plot()

def equalweighting(df_returns, exclude):
    # Get the assets by excluding the specified column
    assets = df_returns.columns[df_returns.columns != exclude]
    # Calculate equal weights for the assets
    weights = np.array([1/len(assets)]*len(assets))

    # Calculate and add the portfolio returns to df_returns
    strategy = df_returns
    strategy['portfolio'] = df_returns[assets].mul(weights, axis=1).sum(axis=1)
    return strategy

eqw = [equalweighting(df_returns.copy(), None)]

def plot_performance(strategy_list=None, portfolio='portfolio', variant=None):
    # Plot cumulative returns
    fig, ax = plt.subplots()
    
    if strategy_list!=None:
        SPY = df['SPY'][strategy_list[0][1].sum(1)>0]
        (SPY/SPY[0]).plot(ax=ax, label='SPY')
        (1+eqw[0][portfolio][strategy_list[0][1].sum(1)>0][1:]).cumprod().plot(ax=ax, label='equal_weight')
        (1+rp[0][0][portfolio][strategy_list[0][1].sum(1)>0][1:]).cumprod().plot(ax=ax, label='risk_parity')
        for i, strategy in enumerate(strategy_list):
            (1+strategy[0][portfolio][strategy_list[0][1].sum(1)>0]).cumprod().plot(ax=ax, label=f'strategy {i+1}')
            if(variant != None):
                for var in variant:
                    (1+strategy[0][var][strategy_list[0][1].sum(1)>0]).cumprod().plot(ax=ax, label=f'strategy {i+1} Variant')
    else:
        SPY = df['SPY']
        (SPY/SPY[0]).plot(ax=ax, label='SPY')
        (1+eqw[0][portfolio]).cumprod().plot(ax=ax, label='equal_weight')
        (1+rp[0][0][portfolio]).cumprod().plot(ax=ax, label='risk_parity')
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    plt.show()
    return None

def plot_histogram(profit):
    #bins = [-0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02]
    mean = profit.mean()
    median = profit.median()
    q1 = profit.quantile(0.25)
    q3 = profit.quantile(0.75)
    std = profit.std()
    # Plot histogram
    fig, ax = plt.subplots()
    profit.hist(ax=ax, bins = 80, edgecolor='black', alpha=0.5)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {"%.4f" % mean}')
    plt.axvline(median, color='m', linestyle='dashed', linewidth=1, label=f'Median = {"%.4f" % median}')
    plt.axvline(q1, color='g', linestyle='dashed', linewidth=1, label=f'Q1 = {"%.4f" % q1}')
    plt.axvline(q3, color='b', linestyle='dashed', linewidth=1, label=f'Q3 = {"%.4f" % q3}')
    plt.axvline(std, color='w', linestyle='dashed', linewidth=1, label=f'Std = {"%.4f" % std}')
    plt.legend()
    
def plot_allocation(df_weights):
    # Assuming monthly rebalancing, forward fill the weights
    df_weights = df_weights.fillna(0).ffill()
    df_weights[df_weights < 0] = 0

    # Plotting
    fig, ax = plt.subplots()
    df_weights.plot.area(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Allocation')
    ax.set_title('Asset Allocation Over Time')
    plt.show()
    return None


def risk_parity(df, df_returns, lookback, period=25):
    # Initialize an empty DataFrame to store the weights
    weights = pd.DataFrame(index=df.index, columns=df.columns)
    strategy = df_returns.copy()
    strategy['portfolio'] = pd.Series(index=df.index, data=np.nan)
    
    # Loop over the DataFrame row by row
    for i in range(lookback + 1, len(df), period):
        # Calculate the standard deviation of the returns for the lookback period
        std_devs = df_returns.iloc[i-lookback:i].std()

        # Calculate the inverse of the standard deviation
        inv_std_devs = 1 / std_devs
        # Normalize the inverse standard deviations to get the portfolio weights
        weights.iloc[i] = inv_std_devs / np.sum(inv_std_devs)
        
    weights.ffill(inplace=True)
    
    for j in range(len(df)-1):
        strategy.iloc[j+1, strategy.columns.get_loc('portfolio')] = np.sum(weights.iloc[j] * df_returns.iloc[j+1])
    
    return strategy, weights

rp = [risk_parity(df.copy(), df_returns.copy(), 25)]
# plot_performance()

def min_var(R, rho=0.001):
    cov_matrix = R.cov().values
    avg_returns = R.mean().values
    n = len(R.columns)
    m = gp.Model("portfolio")
    w = m.addMVar(n, name="w")
    portfolio_variance = (w @ cov_matrix) @ w
    m.setObjective(portfolio_variance, gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, "budget")
    m.addConstr(gp.quicksum(w[i] * avg_returns[i] for i in range(n)) >= rho, "target_return")
    m.optimize()
    # Extract the solution
    solution = []
    for i in range(n):
        print(f"w {i} = {m.getVarByName('w[%d]'%(i)).X}")
        solution.append(m.getVarByName('w[%d]'%(i)).X)
    return solution

############################################################################################
#from multiprocessing import Pool, cpu_count
#from concurrent.futures import ProcessPoolExecutor
#import ipyparallel as ipp

def g(x, phi_star, lambda_star):
    g = x + 2 * (np.outer(x, x) @ phi_star - [np.dot(phi_star, x) * np.dot(x, phi_star)] * len(x)) / lambda_star[0]
    return g

def cal_delta_alpha(R, lookback, rho = 0.02):
    mu = R.mean()                                     #E(R)
    Sigma = R.cov() + np.outer(R.mean(), R.mean())
    Sigma_Inv = pd.DataFrame(np.linalg.pinv(Sigma.values), Sigma.columns, Sigma.index)
    l_0 = 1 - np.dot(mu, (Sigma_Inv @ mu))
    Sigma = Sigma
    Sigma['lambda_1'] = mu
    Sigma['lambda_2'] = -1
    Sigma.loc['lambda_1'] = np.append(mu, [0,0])
    Sigma.loc['lambda_2'] = np.append([1]*len(mu), [0, 0])
    #print(Sigma)
    
    Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), Sigma.columns, Sigma.index) #R^-1
    #print(Sigma_inv.values)
    #print(Sigma.values @ Sigma_inv.values)
    sol_lagrange = np.dot(Sigma_inv, np.append([0]*len(mu), [0.1, 1])) #phi* and lambda*
    phi_star = np.split(sol_lagrange, [len(mu)])[0]
    lambda_star = np.split(sol_lagrange, [len(mu)])[1]
    
    #print(f'sol_lagrange: {sol_lagrange} split into phi*: {phi_star} and lambda*: {lambda_star} \n')
    
    G = np.zeros((lookback, len(mu)))
    for i in range(lookback):
        #print(f'i={i}')
        g_sum = [0] * len(mu)
        for j in range(1, i+2):
            #print(j)
            x = R.iloc[-j].to_numpy()
            g_sum = g_sum + g(x, phi_star, lambda_star)
        G[i] = g_sum/np.sqrt(i+1)
    
    #print(f'G={G}')
    #print(np.std(G, axis=0))
    
    samples = 500000
    Z = np.zeros((len(mu), samples))
    #print(Z)
    for i in range(len(mu)):
        #print(f"find {i}")
        Z[i] = np.random.normal(0, np.std(G, axis=0)[i], samples)
    #print(f'Z={Z}')
    #print(f'|Z|={(np.linalg.norm(Z, axis=0)) ** 2}')
    
    #print(f'{l_0}')
    Rn = ((np.linalg.norm(Z, axis=0)) ** 2) / (4*(lookback)*l_0)
    #print(f'Rn={Rn}')
    
    delta=np.percentile(Rn, 95)
    delta=delta/1000000
    
    G_alpha = np.zeros(lookback)
    for i in range(lookback):
        #print(f'i={i}')
        r_sum = 0
        for j in range(1, i+2):
            #print(j)
            x = R.iloc[-j].to_numpy()
            r_sum = r_sum + np.dot(phi_star, x)/np.linalg.norm(phi_star)
        G_alpha[i] = r_sum/np.sqrt(i+1)
    #print(G_alpha)
    
    Z_alpha = np.zeros(samples)
    Z_alpha = np.random.normal(0, np.std(G_alpha), samples)
    #print(f'Z_alpha={Z_alpha}')
    v_0 = np.percentile(1-Z_alpha/np.sqrt(lookback)/np.sqrt(delta), 95)
    #print(f'v_0={v_0}')
    alpha = rho - np.sqrt(delta) * np.linalg.norm(phi_star) * v_0
    
    return delta, alpha

def dro_drmv_final_decision(R, lookback, delta, alpha_0):
    alpha = 0
    mu = R.mean().values
    sigma = R.std().values
    Sigma = R.cov() #+ np.outer(R.mean(), R.mean())
    Sigma = Sigma.values
    R_neg = R.clip(upper = 0)
    Sigma_neg = R_neg.cov()
    Sigma_neg = Sigma_neg.values
    #print(Sigma)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env, name = "non_linear_convex") as opt_mod:
            #opt_mod = gp.Model()
            opt_mod.params.NonConvex = 2
            opt_mod.setParam('TimeLimit', 30)
    
            phi = opt_mod.addMVar(len(mu), ub=float('inf'), name='phi')
            phi_abs = opt_mod.addMVar(len(mu), name='phi abs')
            y = opt_mod.addVar(name = f"var y", vtype = gp.GRB.CONTINUOUS, lb = 0)
            y2 = opt_mod.addVar(name = f"var y2", vtype = gp.GRB.CONTINUOUS, lb = 0)
            z = opt_mod.addVar(name = f"var z", vtype = gp.GRB.CONTINUOUS, lb = 0)
            z2 = opt_mod.addVar(name = f"var z2", vtype = gp.GRB.CONTINUOUS, lb = 0)
            
            #Objective
            obj_fn = y + np.sqrt(delta) * z
            opt_mod.setObjective(obj_fn, gp.GRB.MINIMIZE)
            
            #Constraint
            c1 = opt_mod.addConstr((phi @ Sigma) @ phi == y2, name = 'c1')
            c2 = opt_mod.addGenConstrPow(xvar=y2, yvar=y, a=0.5, name = 'c2')
            
            c3 = opt_mod.addConstr(phi @ phi == z2, name = 'c3')
            c4 = opt_mod.addGenConstrPow(xvar=z2, yvar=z, a=0.5, name = 'c4')
            c5 = opt_mod.addConstr(phi @ mu >= alpha_0 + np.sqrt(delta) * z, name = 'c5')
            
            c6 = opt_mod.addConstr(phi @ np.ones(len(mu)) == 1, name = 'c6')
            # c6_abs = opt_mod.addConstr(phi_abs @ np.ones(len(mu)) == 1, name = 'c6')
            # for vx, vy in zip(phi.tolist(), phi_abs.tolist()):
            #     opt_mod.addConstr(vy == gp.abs_(vx))
            
            opt_mod.optimize()
            rdobj = 0
            while(opt_mod.status == gp.GRB.INFEASIBLE):
                if rdobj > 40:
                    c5.setAttr(gp.GRB.Attr.RHS, cal_delta_alpha(R, lookback, rho=0.01)[1])
                    opt_mod.optimize()
                    #print(f'failed interrupted: {c5.getAttr(gp.GRB.Attr.RHS)}')
                    rdobj = 0
                    break
                
                #print(f'infeasible: {rdobj}, failed constr: {c5.getAttr(gp.GRB.Attr.RHS)}')
                rate = 0.7
                alpha = alpha_0 * (1-rate) * (rate ** rdobj)
                c5.setAttr(gp.GRB.Attr.RHS, c5.getAttr(gp.GRB.Attr.RHS) - alpha)              
                opt_mod.optimize()
                rdobj = rdobj + 1
            
            #Loosen constraint
            loosen_rate = 0.7
            if rdobj != 0:
                c5.setAttr(gp.GRB.Attr.RHS, c5.getAttr(gp.GRB.Attr.RHS)*loosen_rate)
                opt_mod.optimize()
            
            # if opt_mod.status == gp.GRB.OPTIMAL:
            #     print(f'\nsucceeded constr: {c5.getAttr(gp.GRB.Attr.RHS)}')
            #     print(f'succeeded obj: {y.X} + {np.sqrt(delta)*z.X}')
            #     print("\nOptimal solution found.")
            
            phi_out = []
            for i in range(len(mu)):
                #print(f"phi {i} = {opt_mod.getVarByName('phi[%d]'%(i)).X}")
                phi_out.append(opt_mod.getVarByName('phi[%d]'%(i)).X)
            var_y = opt_mod.getVarByName("var y").x
            #print(f"var y = {var_y}")
            var_z = opt_mod.getVarByName("var z").x
            #print(f"var z = {var_z}")
            #print(f'successful: {c5.getAttr(gp.GRB.Attr.RHS)}')
    
    return phi_out

    
def dro_drmv(df, df_returns, lookback):
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date='2006-02-01', end_date='2029-12-31')
    real_idx = early.groupby(early.index.strftime('%Y-%m')).tail(1).index
    rb_idx = df.groupby(df.index.strftime('%Y-%m')).tail(1).index
    #print(real_idx)
    
    # Initialize an empty DataFrame to store the weights
    weights = pd.DataFrame(index=df.index, columns=df.columns)
    strategy = df_returns.copy()
    strategy['portfolio'] = pd.Series(index=df.index, data=0)
    strategy['portfolio mom'] = pd.Series(index=df.index, data=0)
    #strategy['portfolio sig'] = pd.Series(index=df.index, data=0)
    strategy['portfolio pos'] = pd.Series(index=df.index, data=0)
    df_returns_neg = df_returns.copy()
    df_returns_neg[df_returns_neg > 0] = 0

    # # Loop over the DataFrame row by row
    # for i in range(lookback + 1, len(df), period):
    #     print(i)
    #     print(f'lookback = {lookback}')
    #     R_n = df_returns.iloc[i-lookback:i]
    #     delta, alpha = cal_delta_alpha(R_n, lookback, rho=0.2)
    #     weights.iloc[i] = dro_drmv_final_decision(R_n, lookback, delta, alpha)
    
    # Loop over the rebalance dates
    for rb_date in rb_idx:
        if pd.Timestamp(rb_date.strftime('%Y-%m-%d')) not in real_idx:
            break

        rb_date_index = df.index.get_loc(rb_date)
        
        if rb_date_index > lookback:
            #print(f'\nlookback = {lookback}')
            print(rb_date)
            R_n = df_returns.iloc[rb_date_index - lookback:rb_date_index]
            
            delta, alpha = cal_delta_alpha(R_n, lookback, rho=0.2)
            #print(f'sqrt(delta) = {np.sqrt(delta)}, alpha = {alpha}; no resample')
            
            #Logarithmic Weighting
            # z=np.array(range(lookback))
            # zw = (np.log(z+1)/sum(np.log(z+1))) * lookback
            # R_n = R_n.apply(lambda x: x*zw)
            
            #Resample 1
            # resampled_data = []
            # for idx, row in enumerate(R_n.iterrows()):
            #     for _ in range(idx+1):
            #         resampled_data.append(row[1])
            # R_n = pd.DataFrame(resampled_data, columns=R_n.columns)
            
            #Resample 2
            ##linear interpolation and naming rows with the more recent date
            # resampled_data = [R_n.iloc[0]]
            # resampled_dates = [R_n.index[0]]
            # for idx, (date, row) in enumerate(R_n.iterrows()):
            #     if idx < len(R_n) - 1:  # Exclude last row as there's no next row to interpolate with
            #         next_date, next_row = R_n.index[idx+1], R_n.iloc[idx+1]
            #         for j in range(idx+2):
            #             fraction = (j+1) / (idx+2)
            #             interpolated_row = row + fraction * (next_row - row)
            #             resampled_data.append(interpolated_row)
            #             resampled_dates.append(next_date)
            # R_n = pd.DataFrame(resampled_data, index=resampled_dates, columns=R_n.columns)

            #Big data sample
            # delta, alpha = cal_delta_alpha(R_n, lookback*80, rho=0.2)
            # print(f'sqrt(delta) = {np.sqrt(delta)}, alpha = {alpha}')
            
            weights.loc[rb_date] = dro_drmv_final_decision(R_n, lookback, delta, alpha)
        
    weights.ffill(inplace=True)
    
    #mom = (df.copy()/df.copy().rolling(22*1).mean()-1) + (df.copy()/df.copy().rolling(22*3).mean()-1) + (df.copy()/df.copy().rolling(22*6).mean()-1)
    #mom = (df.copy()/df.copy().rolling(200).mean()-1)
    mom = 6*(df.copy()/df.copy().rolling(22*1).mean()-1) + 3*(df.copy()/df.copy().rolling(22*3).mean()-1) + 1*(df.copy()/df.copy().rolling(22*6).mean()-1)
    #mom = 12*(df.copy()/df.copy().rolling(22*1).mean()-1) + 6*(df.copy()/df.copy().rolling(22*3).mean()-1) + 3*(df.copy()/df.copy().rolling(22*6).mean()-1) + 1*(df.copy()/df.copy().rolling(22*12).mean()-1)
    weights_mom = weights*(mom>=0).astype(int).replace(0,0.5).shift(1)
    #weights_mom = weights * 2
    
    # sig = pd.read_csv("OneDrive\桌面\Project\Quant\signal.csv", index_col = "Date")
    # sig["DOUBLEVV"].index = pd.to_datetime(sig["DOUBLEVV"].index)
    # common_dates = sig["DOUBLEVV"].index.intersection(allocation_drmv[0][1].index)
    # sig_aligned = sig["DOUBLEVV"].replace([0,1,2],[1,0.5,0]).reindex(common_dates)
    # weights_aligned = weights.reindex(common_dates)
    # weights_sig = weights_aligned.multiply(sig_aligned.shift(1), axis=0)
    
    # #Resize for weights_sig
    # weights_copy = weights.copy()
    # for date in weights_sig.index[1:]:
    #     weights_copy.loc[date] = weights_sig.loc[date]
    # weights_sig = weights_copy
    
    # weights_pos = weights.copy()
    # for i in stocks:
    #     EMAd = pd.DataFrame(df.ta.ema(50, close=i) - df.ta.ema(200, close=i), columns=['D'])
    #     EMAd10 = EMAd.ta.ema(5, close='D')
    #     tdEMAdd = round(5 * (np.tanh((EMAd10 - EMAd10.shift(1)))))
    #     weights_pos[i] = weights_pos[i] * tdEMAdd
        
    #     plt.plot(tdEMAdd, label=i)
    # plt.legend()
    # plt.show()
    
    weights = weights.fillna(0)
    weights_mom = weights_mom.fillna(0)
    # weights_pos = weights_pos.fillna(0)

    
    for j in range(len(df)-1):
        strategy.iloc[j+1, strategy.columns.get_loc('portfolio')] = np.sum(weights.iloc[j] * df_returns.iloc[j+1])
        strategy.iloc[j+1, strategy.columns.get_loc('portfolio mom')] = np.sum(weights_mom.iloc[j] * df_returns.iloc[j+1])
        #strategy.iloc[j+1, strategy.columns.get_loc('portfolio sig')] = np.sum(weights_sig.iloc[j] * df_returns.iloc[j+1])
        #strategy.iloc[j+1, strategy.columns.get_loc('portfolio pos')] = np.sum(weights_pos.iloc[j] * df_returns.iloc[j+1])
    
    # print(strategy)
    # print(weights)
    # print(weights_mom)
    # print(weights_pos)
    
    
    
    return strategy, weights, weights_mom


# def process_chunk(i, df, df_returns, lookback):
#     R_n = df_returns.iloc[i-lookback:i]
#     delta, alpha = cal_delta_alpha(R_n, lookback)
#     return i, dro_drmv_final_decision(R_n, delta, alpha)

# def dro_drmv(df, df_returns, lookback, period=30):
#     weights = pd.DataFrame(index=df.index, columns=df.columns)
#     strategy = df_returns.copy()
#     strategy['portfolio'] = pd.Series(index=df.index, data=0)
    
#     indices = list(range(lookback + 1, len(df), period))
    
#     # Using multiprocessing's Pool
#     with Pool(processes=cpu_count()) as pool:
#         results = pool.starmap(process_chunk, [(i, df, df_returns, lookback) for i in indices])
    
#     for i, weight in results:
#         weights.iloc[i] = weight
    
#     weights.ffill(inplace=True)
    
#     for j in range(len(df)-1):
#         strategy.iloc[j+1, strategy.columns.get_loc('portfolio')] = np.sum(weights.iloc[j] * df_returns.iloc[j+1])
    
#     return strategy, weights

###########################################################################################

#allocation_drmv = [dro_drmv(df.copy(), df_returns.copy(), lookback=100)]
allocation_drmv = [dro_drmv(df.copy(), df_returns.copy(), lookback=80)]
#allocation_drmv.append(dro_drmv(df.copy(), df_returns.copy(), lookback=100))
#allocation_drmv.append(dro_drmv(df.copy(), df_returns.copy(), lookback=120))
#allocation_drmv.append(dro_drmv(df.copy(), df_returns.copy(), lookback=150))
#allocation_drmv.append(dro_drmv(df.copy(), df_returns.copy(), lookback=200))

POSITRON = allocation_drmv[0][2]
POSITRON['BIL']=1-POSITRON.sum(1)
print(POSITRON)

portfolio = 'portfolio'
variant = ['portfolio mom']
plot_performance(allocation_drmv, portfolio, variant)

# for w in allocation_drmv:
#     plot_allocation(w[1])
#     plot_allocation(w[2])
#     plot_allocation(w[3])

# df_drmv = pd.DataFrame()
# df_drmv['EQW'] = eqw[0]['portfolio']
# df_drmv['RP'] = rp[0][0]['portfolio']
# df_drmv['SPY'] = df_returns['SPY']
# for i, value in enumerate(allocation_drmv):
#     df_drmv[f'DRMV {i+1}'] = value[0]['portfolio']
#     df_drmv[f'DRMVmom {i+1}'] = value[0]['portfolio mom']
#     #df_drmv[f'DRMVsig {i+1}'] = value[0]['portfolio sig']
#     df_drmv[f'DRMVpos {i+1}'] = value[0]['portfolio pos']
    
# df_metrics = pd.DataFrame()
# qs.reports.metrics(df_drmv, mode="full", display=True)
# qs.reports.metrics(df_returns, mode="full", display=True)

# df_drmv_r = df_drmv.rolling(252).apply(lambda x: np.prod(1 + x) - 1)
# for col in df_drmv_r.columns:
#     plot_histogram(df_drmv_r[col])
    
