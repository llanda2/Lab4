# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import dash
from dash.exceptions import PreventUpdate
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME],
)

#  make dataframe from  spreadsheet:
df = pd.read_csv("assets/historic.csv")

MAX_YR = df.Year.max()
MIN_YR = df.Year.min()
START_YR = 2007

# concat pd fixed
df = (
    pd.concat([df, pd.DataFrame([{"Year": MIN_YR - 1}])], ignore_index=True)
    .sort_values("Year", ignore_index=True)
    .fillna(0)
)

# Create history DataFrame to store previous analyses
history_df = pd.DataFrame(columns=[
    'Date', 'Cash %', 'Bonds %', 'Stocks %',
    'Starting Amount', 'Start Year', 'Duration (Years)',
    'Ending Amount', 'CAGR'
])

COLORS = {
    "cash": "#3cb521",
    "bonds": "#fd7e14",
    "stocks": "#446e9b",
    "inflation": "#cd0200",
    "background": "whitesmoke",
}

"""
==========================================================================
Markdown Text
"""

datasource_text = dcc.Markdown(
    """
    [Data source:](http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html)
    Historical Returns on Stocks, Bonds and Bills from NYU Stern School of
    Business
    """
)

asset_allocation_text = dcc.Markdown(
    """
> **Asset allocation** is one of the main factors that drive portfolio risk and returns.   Play with the app and see for yourself!

> Change the allocation to cash, bonds and stocks on the sliders and see how your portfolio performs over time in the graph.
  Try entering different time periods and dollar amounts too.
"""
)

learn_text = dcc.Markdown(
    """
    Past performance certainly does not determine future results, but you can still
    learn a lot by reviewing how various asset classes have performed over time.

    Use the sliders to change the asset allocation (how much you invest in cash vs
    bonds vs stock) and see how this affects your returns.

    Note that the results shown in "My Portfolio" assumes rebalancing was done at
    the beginning of every year.  Also, this information is based on the S&P 500 index
    as a proxy for "stocks", the 10 year US Treasury Bond for "bonds" and the 3 month
    US Treasury Bill for "cash."  Your results of course,  would be different based
    on your actual holdings.

    This is intended to help you determine your investment philosophy and understand
    what sort of risks and returns you might see for each asset category.

    The  data is from [Aswath Damodaran](http://people.stern.nyu.edu/adamodar/New_Home_Page/home.htm)
    who teaches  corporate finance and valuation at the Stern School of Business
    at New York University.

    Check out his excellent on-line course in
    [Investment Philosophies.](http://people.stern.nyu.edu/adamodar/New_Home_Page/webcastinvphil.htm)
    """
)

cagr_text = dcc.Markdown(
    """
    (CAGR) is the compound annual growth rate.  It measures the rate of return for an investment over a period of time, 
    such as 5 or 10 years. The CAGR is also called a "smoothed" rate of return because it measures the growth of
     an investment as if it had grown at a steady rate on an annually compounded basis.
    """
)

history_text = dcc.Markdown(
    """
    This tab records a history of your previous analyses. Each time you make changes to your asset allocation, 
    starting amount, or time period, the parameters and results are saved here.

    You can use this history to compare different allocation strategies and see how they impact your returns.
    """
)

footer = html.Div(
    dcc.Markdown(
        """
         This information is intended solely as general information for educational
        and entertainment purposes only and is not a substitute for professional advice and
        services from qualified financial services providers familiar with your financial
        situation.    
        """
    ),
    className="p-2 mt-5 bg-primary text-white small",
)

"""
==========================================================================
Tables
"""

total_returns_table = dash_table.DataTable(
    id="total_returns",
    columns=[{"id": "Year", "name": "Year", "type": "text"}]
            + [
                {"id": col, "name": col, "type": "numeric", "format": {"specifier": "$,.0f"}}
                for col in ["Cash", "Bonds", "Stocks", "Total"]
            ],
    page_size=15,
    style_table={"overflowX": "scroll"},
)

annual_returns_pct_table = dash_table.DataTable(
    id="annual_returns_pct",
    columns=(
            [{"id": "Year", "name": "Year", "type": "text"}]
            + [
                {"id": col, "name": col, "type": "numeric", "format": {"specifier": ".1%"}}
                for col in df.columns[1:]
            ]
    ),
    data=df.to_dict("records"),
    sort_action="native",
    page_size=15,
    style_table={"overflowX": "scroll"},
)

# Create history table
history_table = dash_table.DataTable(
    id="history_table",
    columns=[
        {"id": "Date", "name": "Date", "type": "text"},
        {"id": "Cash %", "name": "Cash %", "type": "numeric", "format": {"specifier": ".0f"}},
        {"id": "Bonds %", "name": "Bonds %", "type": "numeric", "format": {"specifier": ".0f"}},
        {"id": "Stocks %", "name": "Stocks %", "type": "numeric", "format": {"specifier": ".0f"}},
        {"id": "Starting Amount", "name": "Starting Amount", "type": "numeric", "format": {"specifier": "$,.0f"}},
        {"id": "Start Year", "name": "Start Year", "type": "numeric"},
        {"id": "Duration (Years)", "name": "Duration (Years)", "type": "numeric"},
        {"id": "Ending Amount", "name": "Ending Amount", "type": "numeric", "format": {"specifier": "$,.0f"}},
        {"id": "CAGR", "name": "CAGR", "type": "text"},
    ],
    data=history_df.to_dict("records"),
    sort_action="native",
    page_size=15,
    style_table={"overflowX": "scroll"},
)


def make_summary_table(dff):
    """Make html table to show cagr and  best and worst periods"""

    table_class = "h5 text-body text-nowrap"
    cash = html.Span(
        [html.I(className="fa fa-money-bill-alt"), " Cash"], className=table_class
    )
    bonds = html.Span(
        [html.I(className="fa fa-handshake"), " Bonds"], className=table_class
    )
    stocks = html.Span(
        [html.I(className="fa fa-industry"), " Stocks"], className=table_class
    )
    inflation = html.Span(
        [html.I(className="fa fa-ambulance"), " Inflation"], className=table_class
    )

    start_yr = dff["Year"].iat[0]
    end_yr = dff["Year"].iat[-1]

    df_table = pd.DataFrame(
        {
            "": [cash, bonds, stocks, inflation],
            f"Rate of Return (CAGR) from {start_yr} to {end_yr}": [
                cagr(dff["all_cash"]),
                cagr(dff["all_bonds"]),
                cagr(dff["all_stocks"]),
                cagr(dff["inflation_only"]),
            ],
            f"Worst 1 Year Return": [
                worst(dff, "3-mon T.Bill"),
                worst(dff, "10yr T.Bond"),
                worst(dff, "S&P 500"),
                "",
            ],
        }
    )
    return dbc.Table.from_dataframe(df_table, bordered=True, hover=True)


"""
==========================================================================
Figures
"""


def make_bar_chart(slider_input, title):
    """Create a bar chart for asset allocation instead of a pie chart"""
    categories = ["Cash", "Bonds", "Stocks"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=slider_input,
            text=[f"{val}%" for val in slider_input],
            textposition='auto',
            marker_color=[COLORS["cash"], COLORS["bonds"], COLORS["stocks"]],
            width=0.6,
        )
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(b=25, t=75, l=35, r=25),
        height=325,
        paper_bgcolor=COLORS["background"],
        yaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            dtick=20
        ),
        xaxis=dict(
            title="Asset Type"
        )
    )
    return fig


def make_line_chart(dff):
    start = dff.loc[1, "Year"]
    yrs = dff["Year"].size - 1
    dtick = 1 if yrs < 16 else 2 if yrs in range(16, 30) else 5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_cash"],
            name="All Cash",
            marker_color=COLORS["cash"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_bonds"],
            name="All Bonds (10yr T.Bonds)",
            marker_color=COLORS["bonds"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_stocks"],
            name="All Stocks (S&P500)",
            marker_color=COLORS["stocks"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["Total"],
            name="My Portfolio",
            marker_color="black",
            line=dict(width=6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["inflation_only"],
            name="Inflation",
            visible=True,
            marker_color=COLORS["inflation"],
        )
    )
    fig.update_layout(
        title=f"Returns for {yrs} years starting {start}",
        template="none",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(tickprefix="$", fixedrange=True),
        xaxis=dict(title="Year Ended", fixedrange=True, dtick=dtick),
    )
    return fig


"""
==========================================================================
Make Tabs
"""

# =======Play tab components

asset_allocation_card = dbc.Card(asset_allocation_text, className="mt-2")

slider_card = dbc.Card(
    [
        html.H4("First set cash allocation %:", className="card-title"),
        dcc.Slider(
            id="cash",
            marks={i: f"{i}%" for i in range(0, 101, 10)},
            min=0,
            max=100,
            step=5,
            value=10,
            included=False,
        ),
        html.H4(
            "Then set stock allocation % ",
            className="card-title mt-3",
        ),
        html.Div("(The rest will be bonds)", className="card-title"),
        dcc.Slider(
            id="stock_bond",
            marks={i: f"{i}%" for i in range(0, 91, 10)},
            min=0,
            max=90,
            step=5,
            value=50,
            included=False,
        ),
    ],
    body=True,
    className="mt-4",
)
time_period_data = [
    {
        "label": f"2007-2008: Great Financial Crisis to {MAX_YR}",
        "start_yr": 2007,
        "planning_time": MAX_YR - START_YR + 1,
    },
    {
        "label": "1999-2010: The decade including 2000 Dotcom Bubble peak",
        "start_yr": 1999,
        "planning_time": 10,
    },
    {
        "label": "1969-1979:  The 1970s Energy Crisis",
        "start_yr": 1970,
        "planning_time": 10,
    },
    {
        "label": "1929-1948:  The 20 years following the start of the Great Depression",
        "start_yr": 1929,
        "planning_time": 20,
    },
    {
        "label": f"{MIN_YR}-{MAX_YR}",
        "start_yr": "1928",
        "planning_time": MAX_YR - MIN_YR + 1,
    },
]

time_period_card = dbc.Card(
    [
        html.H4(
            "Or select a time period:",
            className="card-title",
        ),
        dbc.RadioItems(
            id="time_period",
            options=[
                {"label": period["label"], "value": i}
                for i, period in enumerate(time_period_data)
            ],
            value=0,
            labelClassName="mb-2",
        ),
    ],
    body=True,
    className="mt-4",
)

# ======= InputGroup components

start_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Amount $"),
        dbc.Input(
            id="starting_amount",
            placeholder="Min $10",
            type="number",
            min=10,
            value=10000,
        ),
    ],
    className="mb-3",
)
start_year = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Year"),
        dbc.Input(
            id="start_yr",
            placeholder=f"min {MIN_YR}   max {MAX_YR}",
            type="number",
            min=MIN_YR,
            max=MAX_YR,
            value=START_YR,
        ),
    ],
    className="mb-3",
)
number_of_years = dbc.InputGroup(
    [
        dbc.InputGroupText("Number of Years:"),
        dbc.Input(
            id="planning_time",
            placeholder="# yrs",
            type="number",
            min=1,
            value=MAX_YR - START_YR + 1,
        ),
    ],
    className="mb-3",
)
end_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Ending Amount"),
        dbc.Input(id="ending_amount", disabled=True, className="text-black"),
    ],
    className="mb-3",
)
rate_of_return = dbc.InputGroup(
    [
        dbc.InputGroupText(
            "Rate of Return(CAGR)",
            id="tooltip_target",
            className="text-decoration-underline",
        ),
        dbc.Input(id="cagr", disabled=True, className="text-black"),
        dbc.Tooltip(cagr_text, target="tooltip_target"),
    ],
    className="mb-3",
)

# Add bonds display component
bonds_display = dbc.InputGroup(
    [
        dbc.InputGroupText("Bonds Allocation:"),
        dbc.Input(id="bonds-allocation", disabled=True, className="text-black"),
    ],
    className="mb-3",
)

input_groups = html.Div(
    [start_amount, start_year, number_of_years, end_amount, rate_of_return, bonds_display],
    className="mt-4 p-4",
)
# Add Previous Setting button next to the input groups
previous_setting_button = dbc.Button(
    "Previous Setting",
    id="previous-setting-button",
    color="secondary",
    className="mt-2 mb-3",
    disabled=True
)

input_groups_with_button = html.Div(
    [html.Div([input_groups, previous_setting_button])],
    className="mt-4"
)
# =====  Results Tab components

results_card = dbc.Card(
    [
        dbc.CardHeader("My Portfolio Returns - Rebalanced Annually"),
        html.Div(total_returns_table),
    ],
    className="mt-4",
)

data_source_card = dbc.Card(
    [
        dbc.CardHeader("Source Data: Annual Total Returns"),
        html.Div(annual_returns_pct_table),
    ],
    className="mt-4",
)

# ========= Learn Tab Components
learn_card = dbc.Card(
    [
        dbc.CardHeader("An Introduction to Asset Allocation"),
        dbc.CardBody(learn_text),
    ],
    className="mt-4",
)

# ========= History Tab Components
history_card = dbc.Card(
    [
        dbc.CardHeader("Analysis History"),
        dbc.CardBody([
            history_text,
            html.Div(history_table),
            dbc.Button("Clear History", id="clear-history", color="primary", className="mt-3"),
        ]),
    ],
    className="mt-4",
)

# ========= Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(learn_card, tab_id="tab1", label="Learn"),
        dbc.Tab(
            [asset_allocation_text, slider_card, input_groups_with_button, time_period_card],
            tab_id="tab-2",
            label="Play",
            className="pb-4",
        ),
        dbc.Tab([results_card, data_source_card], tab_id="tab-3", label="Results"),
        dbc.Tab(history_card, tab_id="tab-4", label="History"),
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)

"""
==========================================================================
Helper functions to calculate investment results, cagr and worst periods
"""


def backtest(stocks, cash, start_bal, nper, start_yr):
    """calculates the investment returns for user selected asset allocation,
    rebalanced annually and returns a dataframe
    """

    end_yr = start_yr + nper - 1
    cash_allocation = cash / 100
    stocks_allocation = stocks / 100
    bonds_allocation = (100 - stocks - cash) / 100

    # Select time period - since data is for year end, include year prior
    # for start ie year[0]
    dff = df[(df.Year >= start_yr - 1) & (df.Year <= end_yr)].set_index(
        "Year", drop=False
    )
    dff["Year"] = dff["Year"].astype(int)

    # add columns for My Portfolio returns
    dff["Cash"] = cash_allocation * start_bal
    dff["Bonds"] = bonds_allocation * start_bal
    dff["Stocks"] = stocks_allocation * start_bal
    dff["Total"] = start_bal
    dff["Rebalance"] = True

    # calculate My Portfolio returns
    for yr in dff.Year + 1:
        if yr <= end_yr:
            # Rebalance at the beginning of the period by reallocating
            # last period's total ending balance
            if dff.loc[yr, "Rebalance"]:
                dff.loc[yr, "Cash"] = dff.loc[yr - 1, "Total"] * cash_allocation
                dff.loc[yr, "Stocks"] = dff.loc[yr - 1, "Total"] * stocks_allocation
                dff.loc[yr, "Bonds"] = dff.loc[yr - 1, "Total"] * bonds_allocation

            # calculate this period's  returns
            dff.loc[yr, "Cash"] = dff.loc[yr, "Cash"] * (
                    1 + dff.loc[yr, "3-mon T.Bill"]
            )
            dff.loc[yr, "Stocks"] = dff.loc[yr, "Stocks"] * (1 + dff.loc[yr, "S&P 500"])
            dff.loc[yr, "Bonds"] = dff.loc[yr, "Bonds"] * (
                    1 + dff.loc[yr, "10yr T.Bond"]
            )
            dff.loc[yr, "Total"] = int(dff.loc[yr, ["Cash", "Bonds", "Stocks"]].sum())

    dff = dff.reset_index(drop=True)
    columns = ["Cash", "Stocks", "Bonds", "Total"]
    dff[columns] = dff[columns].round(0)

    # create columns for when portfolio is all cash, all bonds or  all stocks,
    #   include inflation too
    #
    # create new df that starts in yr 1 rather than yr 0
    dff1 = (dff[(dff.Year >= start_yr) & (dff.Year <= end_yr)]).copy()
    #
    # calculate the returns in new df:
    columns = ["all_cash", "all_bonds", "all_stocks", "inflation_only"]
    annual_returns = ["3-mon T.Bill", "10yr T.Bond", "S&P 500", "Inflation"]
    for col, return_pct in zip(columns, annual_returns):
        dff1[col] = round(start_bal * (1 + (1 + dff1[return_pct]).cumprod() - 1), 0)
    #
    # select columns in the new df to merge with original
    dff1 = dff1[["Year"] + columns]
    dff = dff.merge(dff1, how="left")
    # fill in the starting balance for year[0]
    dff.loc[0, columns] = start_bal
    return dff


def cagr(dff):
    """calculate Compound Annual Growth Rate for a series and returns a formated string"""

    start_bal = dff.iat[0]
    end_bal = dff.iat[-1]
    planning_time = len(dff) - 1
    cagr_result = ((end_bal / start_bal) ** (1 / planning_time)) - 1
    return f"{cagr_result:.1%}"


def worst(dff, asset):
    """calculate worst returns for asset in selected period returns formated string"""

    worst_yr_loss = min(dff[asset])
    worst_yr = dff.loc[dff[asset] == worst_yr_loss, "Year"].iloc[0]
    return f"{worst_yr_loss:.1%} in {worst_yr}"


"""
===========================================================================
Main Layout
"""

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Asset Allocation Visualizer",
                    className="text-center bg-primary text-white p-2",
                ),
            )
        ),  # Added subheading with name and course information
        dbc.Row(
            dbc.Col(
                html.H5(
                    "Created by: Lauren Landa for CS-150",
                    className="text-center p-1",
                ),
            )
        ),
        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
                        dcc.Graph(id="allocation_bar_chart", className="mb-2"),
                        dcc.Graph(id="returns_chart", className="pb-4"),
                        html.Hr(),
                        html.Div(id="summary_table"),
                        html.H6(datasource_text, className="my-2"),
                    ],
                    width=12,
                    lg=7,
                    className="pt-4",
                ),
            ],
            className="ms-1",
        ),
        dbc.Row(dbc.Col(footer)),
        # Store for the history data
        dcc.Store(id='history-data', data=history_df.to_dict('records')),
        # Store for tracking the current position in history
        dcc.Store(id='history-position', data=0),
    ],
    fluid=True,
)

"""
==========================================================================
Callbacks
"""


@app.callback(
    Output("bonds-allocation", "value"),
    Input("stock_bond", "value"),
    Input("cash", "value"),
)
def update_bonds_display(stocks, cash):
    bonds = 100 - stocks - cash
    return f"{bonds}%"


@app.callback(
    Output("allocation_bar_chart", "figure"),
    Input("stock_bond", "value"),
    Input("cash", "value"),
)
def update_bar_chart(stocks, cash):
    bonds = 100 - stocks - cash
    slider_input = [cash, bonds, stocks]

    if stocks >= 70:
        investment_style = "Aggressive"
    elif stocks <= 30:
        investment_style = "Conservative"
    else:
        investment_style = "Moderate"
    figure = make_bar_chart(slider_input, investment_style + " Asset Allocation")
    return figure


@app.callback(
    Output("stock_bond", "max"),
    Output("stock_bond", "marks"),
    Output("stock_bond", "value"),
    Input("cash", "value"),
    State("stock_bond", "value"),
)
def update_stock_slider(cash, initial_stock_value):
    max_slider = 100 - int(cash)
    stocks = min(max_slider, initial_stock_value)

    # formats the slider scale
    if max_slider > 50:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 10)}
    elif max_slider <= 15:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 1)}
    else:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 5)}
    return max_slider, marks_slider, stocks


@app.callback(
    Output("planning_time", "value"),
    Output("start_yr", "value"),
    Output("time_period", "value"),
    Input("planning_time", "value"),
    Input("start_yr", "value"),
    Input("time_period", "value"),
)
def update_time_period(planning_time, start_yr, period_number):
    """syncs inputs and selected time periods"""
    ctx = callback_context
    input_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if input_id == "time_period":
        planning_time = time_period_data[period_number]["planning_time"]
        start_yr = time_period_data[period_number]["start_yr"]

    if input_id in ["planning_time", "start_yr"]:
        period_number = None

    return planning_time, start_yr, period_number


@app.callback(
    Output("total_returns", "data"),
    Output("returns_chart", "figure"),
    Output("summary_table", "children"),
    Output("ending_amount", "value"),
    Output("cagr", "value"),
    Output('history-data', 'data'),
    Input("stock_bond", "value"),
    Input("cash", "value"),
    Input("starting_amount", "value"),
    Input("planning_time", "value"),
    Input("start_yr", "value"),
    State('history-data', 'data')
)
def update_totals(stocks, cash, start_bal, planning_time, start_yr, history_data):
    # set defaults for invalid inputs
    start_bal = 10 if start_bal is None else start_bal
    planning_time = 1 if planning_time is None else planning_time
    start_yr = MIN_YR if start_yr is None else int(start_yr)

    # calculate valid planning time start yr
    max_time = MAX_YR + 1 - start_yr
    planning_time = min(max_time, planning_time)
    if start_yr + planning_time > MAX_YR:
        start_yr = min(df.iloc[-planning_time, 0], MAX_YR)  # 0 is Year column

    # create investment returns dataframe
    dff = backtest(stocks, cash, start_bal, planning_time, start_yr)

    # create data for DataTable
    data = dff.to_dict("records")

    # create the line chart
    fig = make_line_chart(dff)

    summary_table = make_summary_table(dff)

    # format ending balance
    ending_amount = f"${dff['Total'].iloc[-1]:0,.0f}"
    ending_amount_numeric = dff['Total'].iloc[-1]

    # calculate cagr
    ending_cagr = cagr(dff["Total"])

    # Only add to history if triggered by user action
    ctx = callback_context
    if ctx.triggered:
        # Load existing history data
        history_df = pd.DataFrame(history_data)

        # Add new record to history
        bonds = 100 - stocks - cash
        new_record = pd.DataFrame([{
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Cash %': cash,
            'Bonds %': bonds,
            'Stocks %': stocks,
            'Starting Amount': start_bal,
            'Start Year': start_yr,
            'Duration (Years)': planning_time,
            'Ending Amount': ending_amount_numeric,
            'CAGR': ending_cagr
        }])

        # Concatenate with existing history
        if not history_df.empty:
            history_df = pd.concat([history_df, new_record], ignore_index=True)
        else:
            history_df = new_record

        # Update history data
        history_data = history_df.to_dict('records')

    return data, fig, summary_table, ending_amount, ending_cagr, history_data

@app.callback(
    Output('history_table', 'data'),
    Output('history-position', 'data', allow_duplicate=True),
    Input('history-data', 'data'),
    Input('clear-history', 'n_clicks'),
    State('history-position', 'data'),
    prevent_initial_call=True
)
def update_history_table(history_data, clear_clicks, position):
    ctx = callback_context
    if ctx.triggered_id == 'clear-history' and clear_clicks:
        # Clear history if button was clicked
        return [], 0

    # Otherwise, update with current history data
    return history_data, dash.no_update

@app.callback(
    Output('previous-setting-button', 'disabled'),
    Output('history-position', 'data'),
    Output('cash', 'value', allow_duplicate=True),
    Output('stock_bond', 'value', allow_duplicate=True),
    Output('starting_amount', 'value', allow_duplicate=True),
    Output('start_yr', 'value', allow_duplicate=True),
    Output('planning_time', 'value', allow_duplicate=True),
    Input('previous-setting-button', 'n_clicks'),
    Input('history-data', 'data'),
    State('history-position', 'data'),
    prevent_initial_call=True
)
def recall_previous_setting(n_clicks, history_data, position):
    ctx = callback_context
    triggered_id = ctx.triggered_id if ctx.triggered else None

    # Initialize default values
    disabled = True
    cash_value = dash.no_update
    stock_bond_value = dash.no_update
    starting_amount_value = dash.no_update
    start_yr_value = dash.no_update
    planning_time_value = dash.no_update

    # Only process if we have history data
    if history_data and len(history_data) > 0:
        # Enable button if we have history
        disabled = False

        # If button was clicked, update position and values
        if triggered_id == 'previous-setting-button' and n_clicks:
            # Update position - move one step back in history
            position = min(position + 1, len(history_data) - 1)

            # If we're not out of history
            if position < len(history_data):
                entry = history_data[position]
                cash_value = entry['Cash %']
                stock_bond_value = entry['Stocks %']
                starting_amount_value = entry['Starting Amount']
                start_yr_value = entry['Start Year']
                planning_time_value = entry['Duration (Years)']

            # If we've reached the end of history, disable the button
            if position >= len(history_data) - 1:
                disabled = True

        # When history is cleared, reset position
        elif triggered_id == 'history-data' and len(history_data) == 0:
            position = 0
            disabled = True

    @app.callback(
        Output('previous-setting-button', 'disabled'),
        Output('history-position', 'data'),
        Output('cash', 'value', allow_duplicate=True),
        Output('stock_bond', 'value', allow_duplicate=True),
        Output('starting_amount', 'value', allow_duplicate=True),
        Output('start_yr', 'value', allow_duplicate=True),
        Output('planning_time', 'value', allow_duplicate=True),
        Output('history_table', 'data'),
        Input('previous-setting-button', 'n_clicks'),
        Input('clear-history', 'n_clicks'),
        Input('history-data', 'data'),
        State('history-position', 'data'),
        prevent_initial_call=True
    )
    def manage_history(prev_n_clicks, clear_n_clicks, history_data, position):
        ctx = callback_context
        triggered_id = ctx.triggered_id if ctx.triggered else None

        # Initialize default values
        disabled = True
        cash_value = dash.no_update
        stock_bond_value = dash.no_update
        starting_amount_value = dash.no_update
        start_yr_value = dash.no_update
        planning_time_value = dash.no_update
        history_table_data = history_data

        # If we have history, enable the button
        if history_data and len(history_data) > 0:
            disabled = False

        # Handle the Previous Setting button
        if triggered_id == 'previous-setting-button' and prev_n_clicks:
            if history_data and position < len(history_data):
                # Move one step back in history
                position = min(position + 1, len(history_data) - 1)

                # Apply the historical settings
                entry = history_data[position]
                cash_value = entry['Cash %']
                stock_bond_value = entry['Stocks %']
                starting_amount_value = entry['Starting Amount']
                start_yr_value = entry['Start Year']
                planning_time_value = entry['Duration (Years)']

                # If we've reached the end of history, disable the button
                if position >= len(history_data) - 1:
                    disabled = True

        # Handle the Clear History button
        elif triggered_id == 'clear-history' and clear_n_clicks:
            history_table_data = []
            position = 0
            disabled = True

        # When new history is added, reset position and enable button
        elif triggered_id == 'history-data':
            # Only reset position when new history is added (not when history is cleared)
            if history_data and len(history_data) > 0:
                position = 0
                disabled = False
            else:
                position = 0
                disabled = True

        return disabled, position, cash_value, stock_bond_value, starting_amount_value, start_yr_value, planning_time_value, history_table_data
    return disabled, position, cash_value, stock_bond_value, starting_amount_value, start_yr_value, planning_time_value
if __name__ == "__main__":
    app.run(debug=True)
