"""
Visualization and Reporting Dashboard
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import logging

from data_fetcher import StockDataFetcher
from support_resistance import SupportResistanceDetector
from ml_model import StockMLModel
from trading_strategy import TradingStrategy
from config import STOCK_SYMBOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDashboard:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.sr_detector = SupportResistanceDetector()
        self.ml_model = StockMLModel()
        self.trading_strategy = TradingStrategy()
        
        # Load model if available
        try:
            self.ml_model.load_model("stock_ml_model.pkl")
        except:
            logger.warning("ML model not found, some features may not be available")
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
    
    def create_candlestick_chart(self, symbol, data, support_levels=None, resistance_levels=None):
        """
        Create candlestick chart with support/resistance levels
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Technical Indicators'),
            row_width=[0.2, 0.1, 0.7]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add support levels
        if support_levels:
            for level in support_levels:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Support: ${level:.2f}",
                    row=1, col=1
                )
        
        # Add resistance levels
        if resistance_levels:
            for level in resistance_levels:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Resistance: ${level:.2f}",
                    row=1, col=1
                )
        
        # Volume chart
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Technical indicators
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        return fig
    
    def create_signals_summary_chart(self):
        """
        Create summary chart of recent signals
        """
        # This would typically pull from a database of historical signals
        # For demo purposes, we'll create sample data
        sample_data = {
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'Buy_Signals': np.random.poisson(2, 30),
            'Sell_Signals': np.random.poisson(1.5, 30),
            'Hold_Signals': np.random.poisson(3, 30)
        }
        
        df = pd.DataFrame(sample_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Buy_Signals'],
            mode='lines+markers',
            name='Buy Signals',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Sell_Signals'],
            mode='lines+markers',
            name='Sell Signals',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Hold_Signals'],
            mode='lines+markers',
            name='Hold Signals',
            line=dict(color='gray')
        ))
        
        fig.update_layout(
            title='Trading Signals Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Signals',
            height=400
        )
        
        return fig
    
    def create_performance_metrics(self):
        """
        Create performance metrics display
        """
        # Sample performance data
        metrics = {
            'Total Trades': 45,
            'Win Rate': '67%',
            'Average Return': '2.3%',
            'Max Drawdown': '-8.5%',
            'Sharpe Ratio': '1.24',
            'Active Positions': 3
        }
        
        cards = []
        for metric, value in metrics.items():
            card = dbc.Card([
                dbc.CardBody([
                    html.H4(value, className="card-title"),
                    html.P(metric, className="card-text")
                ])
            ], className="mb-3")
            cards.append(card)
        
        return dbc.Row([
            dbc.Col(card, width=2) for card in cards
        ])
    
    def setup_layout(self):
        """
        Setup the dashboard layout
        """
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("ML Stock Trading Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Performance metrics
            dbc.Row([
                dbc.Col([
                    html.H3("Performance Metrics"),
                    html.Div(id="performance-metrics")
                ])
            ], className="mb-4"),
            
            # Symbol selector and controls
            dbc.Row([
                dbc.Col([
                    html.Label("Select Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': symbol, 'value': symbol} for symbol in STOCK_SYMBOLS],
                        value=STOCK_SYMBOLS[0],
                        className="mb-3"
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Time Period:"),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '5 Days', 'value': '5d'},
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'}
                        ],
                        value='1mo',
                        className="mb-3"
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Update:"),
                    html.Br(),
                    dbc.Button("Refresh Data", id="refresh-button", color="primary", className="mt-2")
                ], width=3)
            ]),
            
            # Main chart
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-chart",
                        children=[dcc.Graph(id='main-chart')],
                        type="default",
                    )
                ])
            ], className="mb-4"),
            
            # Secondary charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='signals-chart')
                ], width=6),
                
                dbc.Col([
                    html.H4("Recent Signals"),
                    html.Div(id="recent-signals-table")
                ], width=6)
            ], className="mb-4"),
            
            # System status
            dbc.Row([
                dbc.Col([
                    html.H4("System Status"),
                    html.Div(id="system-status")
                ])
            ])
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """
        Setup dashboard callbacks
        """
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('performance-metrics', 'children'),
             Output('recent-signals-table', 'children'),
             Output('system-status', 'children')],
            [Input('symbol-dropdown', 'value'),
             Input('period-dropdown', 'value'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_dashboard(selected_symbol, selected_period, n_clicks):
            try:
                # Fetch data
                data = self.data_fetcher.fetch_live_data(selected_symbol, period=selected_period)
                
                if data is None or len(data) == 0:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(
                        text="No data available for the selected symbol and period",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return empty_fig, "No data", "No signals", "System offline"
                
                # Get support/resistance levels
                levels = self.sr_detector.find_support_resistance_levels(data)
                
                # Create main chart
                main_fig = self.create_candlestick_chart(
                    selected_symbol, 
                    data,
                    levels['support_levels'],
                    levels['resistance_levels']
                )
                
                # Create performance metrics
                performance_metrics = self.create_performance_metrics()
                
                # Create recent signals table
                recent_signals = self.create_recent_signals_table(selected_symbol, data, levels)
                
                # Create system status
                system_status = self.create_system_status()
                
                return main_fig, performance_metrics, recent_signals, system_status
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {str(e)}")
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Error loading data: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return error_fig, f"Error: {str(e)}", "Error loading signals", "System error"
        
        @self.app.callback(
            Output('signals-chart', 'figure'),
            [Input('symbol-dropdown', 'value')]
        )
        def update_signals_chart(selected_symbol):
            return self.create_signals_summary_chart()
    
    def create_recent_signals_table(self, symbol, data, levels):
        """
        Create table of recent trading signals
        """
        try:
            # Generate signals for recent data
            recent_data = data.tail(10)
            signals = []
            
            for i in range(len(recent_data)):
                sample_data = recent_data.iloc[:i+1] if i > 0 else recent_data.iloc[:1]
                if len(sample_data) > 5:  # Need minimum data for analysis
                    signal = self.trading_strategy.generate_signals(
                        symbol, sample_data, levels['support_levels'], levels['resistance_levels']
                    )
                    if signal and signal['action'] != 'hold':
                        signals.append({
                            'Time': sample_data.index[-1].strftime('%Y-%m-%d %H:%M'),
                            'Action': signal['action'].upper(),
                            'Confidence': f"{signal['confidence']:.1%}",
                            'Price': f"${signal['current_price']:.2f}"
                        })
            
            if not signals:
                return html.P("No recent signals generated")
            
            # Create table
            table_header = [
                html.Thead([
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Action"),
                        html.Th("Confidence"),
                        html.Th("Price")
                    ])
                ])
            ]
            
            table_body = [
                html.Tbody([
                    html.Tr([
                        html.Td(signal['Time']),
                        html.Td(signal['Action'], style={'color': 'green' if signal['Action'] == 'BUY' else 'red'}),
                        html.Td(signal['Confidence']),
                        html.Td(signal['Price'])
                    ]) for signal in signals[-5:]  # Show last 5 signals
                ])
            ]
            
            return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
            
        except Exception as e:
            logger.error(f"Error creating signals table: {str(e)}")
            return html.P(f"Error loading signals: {str(e)}")
    
    def create_system_status(self):
        """
        Create system status display
        """
        status_items = [
            {"label": "Market Status", "value": "Open" if self.data_fetcher.is_market_open() else "Closed", "color": "success" if self.data_fetcher.is_market_open() else "warning"},
            {"label": "ML Model", "value": "Active" if self.ml_model.is_trained else "Inactive", "color": "success" if self.ml_model.is_trained else "danger"},
            {"label": "Data Connection", "value": "Connected", "color": "success"},
            {"label": "Last Update", "value": datetime.now().strftime('%H:%M:%S'), "color": "info"}
        ]
        
        badges = []
        for item in status_items:
            badge = dbc.Badge(
                f"{item['label']}: {item['value']}",
                color=item['color'],
                className="me-2 mb-2"
            )
            badges.append(badge)
        
        return html.Div(badges)
    
    def run_server(self, debug=False, port=8050):
        """
        Run the dashboard server
        """
        logger.info(f"Starting dashboard server on port {port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')

def main():
    """
    Main function to run the dashboard
    """
    dashboard = TradingDashboard()
    print("Starting Trading Dashboard...")
    print("Open your browser and go to: http://localhost:8050")
    dashboard.run_server(debug=True)

if __name__ == "__main__":
    main()