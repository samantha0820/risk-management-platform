"""
Options Pricing and Risk Management System Main Application
Contains interactive dashboard
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import custom modules
from data.data_loader import DataLoader
from data.option_chain import OptionChainLoader
from models.black_scholes import BlackScholes
from models.binomial_tree import BinomialTree
from models.monte_carlo import MonteCarloPricing
from risk.var_calculator import VaRCalculator
import config

# Initialize Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Options Pricing and Risk Management System"

# Initialize components
data_loader = DataLoader()
option_loader = OptionChainLoader()
bs_model = BlackScholes()
binomial_model = BinomialTree()
mc_model = MonteCarloPricing()
var_calculator = VaRCalculator()

# Application layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Options Pricing and Risk Management System", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Control panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Parameter Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Stock Symbol"),
                            dcc.Dropdown(
                                id='symbol-dropdown',
                                options=[{'label': symbol, 'value': symbol} for symbol in config.STOCK_SYMBOLS],
                                value='AAPL'
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Strike Price"),
                            dcc.Input(id='strike-input', type='number', value=100, step=1)
                        ], width=2),
                        dbc.Col([
                            html.Label("Time to Expiry (Years)"),
                            dcc.Input(id='time-input', type='number', value=0.25, step=0.01, min=0.01)
                        ], width=2),
                        dbc.Col([
                            html.Label("Volatility"),
                            dcc.Input(id='volatility-input', type='number', value=0.2, step=0.01, min=0.01)
                        ], width=2),
                        dbc.Col([
                            html.Label("Risk-free Rate"),
                            dcc.Input(id='rate-input', type='number', value=0.05, step=0.001, min=0)
                        ], width=2),
                        dbc.Col([
                            html.Label("Option Type"),
                            dcc.Dropdown(
                                id='option-type-dropdown',
                                options=[
                                    {'label': 'Call Option', 'value': 'call'},
                                    {'label': 'Put Option', 'value': 'put'}
                                ],
                                value='call'
                            )
                        ], width=1)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Main content area
    dbc.Row([
        # Left panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Pricing Results"),
                dbc.CardBody([
                    html.Div(id='pricing-results')
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Greeks"),
                dbc.CardBody([
                    html.Div(id='greeks-results')
                ])
            ])
        ], width=4),
        
        # Right charts
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Price Trend"),
                dbc.CardBody([
                    dcc.Graph(id='stock-price-chart')
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    # Chart area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Option Price Sensitivity Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='sensitivity-chart')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='risk-chart')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Option chain table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Option Chain Data"),
                dbc.CardBody([
                    html.Div(id='option-chain-table')
                ])
            ])
        ])
    ])
], fluid=True)

# Callback functions
@app.callback(
    [Output('pricing-results', 'children'),
     Output('greeks-results', 'children'),
     Output('stock-price-chart', 'figure'),
     Output('sensitivity-chart', 'figure'),
     Output('risk-chart', 'figure'),
     Output('option-chain-table', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('strike-input', 'value'),
     Input('time-input', 'value'),
     Input('volatility-input', 'value'),
     Input('rate-input', 'value'),
     Input('option-type-dropdown', 'value')]
)
def update_results(symbol, strike, time, volatility, rate, option_type):
    """Update all results"""
    
    # Get stock data
    stock_data = data_loader.get_stock_data(symbol)
    current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 100
    
    # Calculate option prices
    if option_type == 'call':
        bs_price = bs_model.price_call(current_price, strike, time, rate, volatility)
        binomial_price = binomial_model.price_option(current_price, strike, time, rate, volatility, 'call')['price']
        mc_price = mc_model.price_european_option(current_price, strike, time, rate, volatility, 'call')['price']
    else:
        bs_price = bs_model.price_put(current_price, strike, time, rate, volatility)
        binomial_price = binomial_model.price_option(current_price, strike, time, rate, volatility, 'put')['price']
        mc_price = mc_model.price_european_option(current_price, strike, time, rate, volatility, 'put')['price']
    
    # Calculate Greeks
    greeks = bs_model.calculate_all_greeks(option_type, current_price, strike, time, rate, volatility)
    
    # Generate pricing results
    pricing_results = html.Div([
        html.H5("Pricing Results"),
        html.P(f"Black-Scholes: ${bs_price:.4f}"),
        html.P(f"Binomial Tree: ${binomial_price:.4f}"),
        html.P(f"Monte Carlo: ${mc_price:.4f}"),
        html.P(f"Current Stock Price: ${current_price:.2f}"),
        html.Hr(),
        html.H6("Model Comparison"),
        html.P(f"BS vs Binomial: {abs(bs_price - binomial_price):.4f}"),
        html.P(f"BS vs MC: {abs(bs_price - mc_price):.4f}")
    ])
    
    # Generate Greeks results
    greeks_results = html.Div([
        html.H5("Greeks"),
        html.P(f"Delta: {greeks['delta']:.4f}"),
        html.P(f"Gamma: {greeks['gamma']:.4f}"),
        html.P(f"Vega: {greeks['vega']:.4f}"),
        html.P(f"Theta: {greeks['theta']:.4f}"),
        html.P(f"Rho: {greeks['rho']:.4f}")
    ])
    
    # Generate stock price chart
    if not stock_data.empty:
        stock_fig = go.Figure()
        stock_fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))
        stock_fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['MA_20'],
            mode='lines',
            name='20-day MA',
            line=dict(color='orange')
        ))
        stock_fig.update_layout(
            title=f"{symbol} Stock Price Trend",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
    else:
        stock_fig = go.Figure()
        stock_fig.add_annotation(text="No Data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Generate sensitivity analysis chart
    S_range = np.linspace(current_price * 0.7, current_price * 1.3, 50)
    prices = []
    for S in S_range:
        if option_type == 'call':
            price = bs_model.price_call(S, strike, time, rate, volatility)
        else:
            price = bs_model.price_put(S, strike, time, rate, volatility)
        prices.append(price)
    
    sensitivity_fig = go.Figure()
    sensitivity_fig.add_trace(go.Scatter(
        x=S_range,
        y=prices,
        mode='lines',
        name=f'{option_type.upper()} Option Price',
        line=dict(color='red')
    ))
    sensitivity_fig.add_vline(x=current_price, line_dash="dash", line_color="gray")
    sensitivity_fig.update_layout(
        title="Option Price vs Stock Price",
        xaxis_title="Stock Price",
        yaxis_title="Option Price",
        height=400
    )
    
    # Generate risk analysis chart
    if not stock_data.empty:
        returns = stock_data['Returns'].dropna()
        var_result = var_calculator.parametric_var(returns)
        
        risk_fig = make_subplots(rows=2, cols=1, subplot_titles=('Return Distribution', 'VaR Analysis'))
        
        # Return distribution
        risk_fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns', opacity=0.7),
            row=1, col=1
        )
        
        # VaR line
        var_value = var_result['var']
        risk_fig.add_vline(x=var_value, line_dash="dash", line_color="red", 
                          annotation_text=f"VaR: {var_value:.4f}", row=1, col=1)
        
        # Rolling VaR
        rolling_var = var_calculator.rolling_var(returns, window=60)
        risk_fig.add_trace(
            go.Scatter(x=rolling_var.index, y=rolling_var, mode='lines', name='Rolling VaR'),
            row=2, col=1
        )
        
        risk_fig.update_layout(height=500, title_text="Risk Analysis")
    else:
        risk_fig = go.Figure()
        risk_fig.add_annotation(text="No Data", xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Generate option chain table
    try:
        option_chain = option_loader.get_option_chain(symbol)
        if not option_chain['calls'].empty:
            # Select first 10 call options
            calls_df = option_chain['calls'].head(10)
            calls_table = dbc.Table.from_dataframe(
                calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']],
                striped=True,
                bordered=True,
                hover=True
            )
        else:
            calls_table = html.P("No option data")
        
        option_chain_content = html.Div([
            html.H6("Call Options"),
            calls_table
        ])
    except:
        option_chain_content = html.P("Unable to fetch option chain data")
    
    return pricing_results, greeks_results, stock_fig, sensitivity_fig, risk_fig, option_chain_content

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
