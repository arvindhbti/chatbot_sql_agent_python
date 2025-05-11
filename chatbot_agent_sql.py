from langchain_openai import AzureChatOpenAI
import os
import pandas as pd
import sqlite3
from tabulate import tabulate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.globals import set_debug
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("azure_endpoint")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("GPT4O_MINI")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["AZURE_OPENAIGPT4O_CHAT_DEPLOYMENT_NAME"] = "GPT-4o"

# Debugging (set to True for debugging)
set_debug(False)


DB_PATH = "/home/ved.deo/crew_agent_approach/hr_database.db"
DB_URI = f"sqlite:///{DB_PATH}"

# Folder to save plots
PLOTS_FOLDER = "hr_plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 120
sns.set_context("talk")

# Database connection pooling
_db_connection = None

def get_db_connection():
    """Get or create a database connection"""
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect(DB_PATH)
        # Enable performance optimizations
        _db_connection.execute("PRAGMA journal_mode = WAL")
        _db_connection.execute("PRAGMA synchronous = NORMAL")
        _db_connection.execute("PRAGMA cache_size = 10000")
    return _db_connection

def close_db_connection():
    """Close the database connection"""
    global _db_connection
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None

def create_database_indexes():
    """Create indexes on commonly queried columns for better performance"""
    conn = get_db_connection()
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_department ON employees(department)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_salary ON employees(salary)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_performance ON employees(performance_rating)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_country ON employees(country)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_title ON employees(job_title)")
        print("Database indexes created or verified.")
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}")

# System prompts
SQL_SYSTEM_PROMPT = """
You are an advanced SQL expert and database analyst specialized in HR data analytics. 
Follow these guidelines to handle complex database queries:

1. For analytical queries, use SQL capabilities like Window Functions, Common Table Expressions (CTEs), and CASE statements.
2. When writing complex queries:
   - Use CTEs for better readability and performance
   - Consider using indexes when filtering (department, salary, performance_rating)
   - Use appropriate JOIN techniques
   - Use subqueries efficiently
3. For statistical analysis:
   - Use aggregate functions (AVG, SUM, COUNT, MIN, MAX)
   - Consider appropriate grouping levels
   - Provide insights on outliers and distributions
4. Format your SQL queries properly with indentation and comments

The database contains an 'employees' table with these columns:
- employee_id: INTEGER (Primary key)
- first_name: TEXT
- last_name: TEXT
- email: TEXT
- gender: TEXT ('Male' or 'Female')
- job_title: TEXT
- department: TEXT (Engineering, Marketing, Sales, HR, Finance)
- salary: INTEGER
- hire_date: TEXT (YYYY-MM-DD format)
- country: TEXT (USA, India, UK, Canada, Germany)
- city: TEXT (various cities in each country)
- performance_rating: INTEGER (1-5 scale)
- years_experience: INTEGER

Always explain your SQL approach after providing results.
"""

VISUALIZATION_SYSTEM_PROMPT = """
You are an advanced HR data visualization specialist. Your job is to create insightful and informative visualizations from HR database query results.

IMPORTANT: When creating visualizations after performing a SQL query:
1. You MUST convert the DataFrame results to a list of dictionaries using df.to_dict('records')
2. ALWAYS pass this converted data as the 'data' parameter to ANY visualization function
3. Do NOT skip this step or the visualization will fail

Example of correct tool usage sequence:
1. First query: `execute_sql_query` to get data
2. Store the result: `sql_result = execute_sql_query(...)`
3. Convert to required format: `data = sql_result.to_dict('records')`
4. Pass to visualization: `create_scatter_plot(data=data, x_column='...', y_column='...', ...)`

Follow these guidelines:
1. First, understand what type of visualization would best represent the data.
2. For categorical data, consider count plots, bar plots, or pie charts.
3. For numerical data, consider histograms, boxplots, scatter plots, or line plots.
4. For relationships, use scatter plots, heatmaps, or grouped bar plots.
5. Always suggest appropriate plot types based on the data structure.
6. For time series or trend data, use line plots.

The HR database contains employee data with these columns:
- employee_id: INTEGER (Primary key)
- first_name: TEXT
- last_name: TEXT
- email: TEXT
- gender: TEXT ('Male' or 'Female')
- job_title: TEXT
- department: TEXT (Engineering, Marketing, Sales, HR, Finance)
- salary: INTEGER
- hire_date: TEXT (YYYY-MM-DD format)
- country: TEXT (USA, India, UK, Canada, Germany)
- city: TEXT (various cities in each country)
- performance_rating: INTEGER (1-5 scale)
- years_experience: INTEGER

First, you should decide what data you need from the database and craft an appropriate SQL query.
Then, choose suitable visualizations to represent the data effectively.
"""

# SQL Tools
@tool
def execute_sql_query(query: str) -> pd.DataFrame:
    """Execute a SQL query against the HR database and return the results as a dataframe.
    
    Args:
        query: SQL query to execute
        
    Returns:
        DataFrame containing query results
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        raise Exception(f"Error executing SQL query: {str(e)}")

# Visualization Tools
@tool
def create_count_plot(data: List[Dict[str, Any]], x_column: str, title: str, 
                    hue: Optional[str] = None, palette: str = "viridis", 
                    orient: str = "v", save: bool = True) -> str:
    """
    Create a count plot (bar chart showing counts of categories).
    
    Args:
        data: List of dictionaries containing the data
        x_column: Column to count
        title: Title for the plot
        hue: Optional column for color grouping
        palette: Color palette (viridis, magma, plasma, inferno, etc.)
        orient: Orientation - 'v' for vertical, 'h' for horizontal
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure()
    
    # Create plot
    if orient == "h":
        ax = sns.countplot(y=x_column, hue=hue, data=df, palette=palette)
        # Add count annotations
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, f'{int(width)}', 
                    ha='left', va='center')
    else:
        ax = sns.countplot(x=x_column, hue=hue, data=df, palette=palette)
        # Add count annotations
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            plt.text(p.get_x() + p.get_width()/2, height + 0.5, f'{int(height)}', 
                    ha='center', va='bottom')
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/countplot_{x_column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_distribution_plot(data: List[Dict[str, Any]], x_column: str, title: str, 
                           hue: Optional[str] = None, kde: bool = True, 
                           palette: str = "viridis", save: bool = True) -> str:
    """
    Create a distribution plot (histogram with optional KDE).
    
    Args:
        data: List of dictionaries containing the data
        x_column: Column to show distribution for
        title: Title for the plot
        hue: Optional column for color grouping
        kde: Whether to show kernel density estimate
        palette: Color palette
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure()
    
    # Create plot
    if hue is not None:
        ax = sns.histplot(data=df, x=x_column, hue=hue, kde=kde, palette=palette)
    else:
        ax = sns.histplot(data=df, x=x_column, kde=kde, color=sns.color_palette(palette)[0])
        
        # Add mean line
        mean_val = df[x_column].mean()
        plt.axvline(x=mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}')
        plt.legend()
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/distplot_{x_column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_boxplot(data: List[Dict[str, Any]], x_column: str, y_column: str, 
                 title: str, hue: Optional[str] = None, palette: str = "viridis",
                 save: bool = True) -> str:
    """
    Create a boxplot showing distribution statistics.
    
    Args:
        data: List of dictionaries containing the data
        x_column: Categorical column for x-axis
        y_column: Numerical column for y-axis
        title: Title for the plot
        hue: Optional column for color grouping
        palette: Color palette
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure()
    
    # Create plot
    ax = sns.boxplot(data=df, x=x_column, y=y_column, hue=hue, palette=palette)
    
    # Add jitter points for better visualization
    sns.stripplot(data=df, x=x_column, y=y_column, hue=hue, palette=palette, 
                 alpha=0.3, dodge=True, ax=ax)
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/boxplot_{x_column}_{y_column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_heatmap(data: List[Dict[str, Any]], columns: List[str], 
                 title: str, corr_method: str = "pearson", 
                 cmap: str = "viridis", annot: bool = True, save: bool = True) -> str:
    """
    Create a correlation heatmap for numerical columns.
    
    Args:
        data: List of dictionaries containing the data
        columns: List of numerical columns to include in correlation
        title: Title for the plot
        corr_method: Correlation method (pearson, kendall, spearman)
        cmap: Colormap
        annot: Whether to annotate cells with correlation values
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Select only the columns for correlation
    df_subset = df[columns]
    
    # Compute correlation matrix
    corr_matrix = df_subset.corr(method=corr_method)
    
    # Create figure
    plt.figure(figsize=(len(columns) + 2, len(columns) + 1))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, 
              vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
              fmt=".2f", cbar_kws={"shrink": .8})
    
    # Set title
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/heatmap_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_scatter_plot(data: List[Dict[str, Any]], x_column: str, y_column: str, 
                      title: str, hue: Optional[str] = None, size: Optional[str] = None,
                      palette: str = "viridis", regression_line: bool = True, 
                      save: bool = True) -> str:
    """
    Create a scatter plot with optional regression line.
    
    Args:
        data: List of dictionaries containing the data
        x_column: Column for x-axis
        y_column: Column for y-axis
        title: Title for the plot
        hue: Optional column for color grouping
        size: Optional column for point size
        palette: Color palette
        regression_line: Whether to add regression line
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure()
    
    # Create plot
    if regression_line:
        ax = sns.regplot(data=df, x=x_column, y=y_column, 
                        scatter_kws={'alpha': 0.5}, 
                        line_kws={'color': 'red'})
        
        # Add correlation text
        corr = df[x_column].corr(df[y_column])
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        
        if hue is not None or size is not None:
            # Add colored points on top of regression plot
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, size=size,
                           palette=palette, ax=ax, alpha=0.7)
    else:
        # Just create scatterplot
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, size=size,
                      palette=palette, alpha=0.7)
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/scatter_{x_column}_{y_column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_bar_plot(data: List[Dict[str, Any]], x_column: str, y_column: str, 
                  title: str, hue: Optional[str] = None, palette: str = "viridis",
                  orient: str = "v", error_bars: bool = False, save: bool = True) -> str:
    """
    Create a bar plot showing averages or sums.
    
    Args:
        data: List of dictionaries containing the data
        x_column: Column for categories
        y_column: Column for values
        title: Title for the plot
        hue: Optional column for color grouping
        palette: Color palette
        orient: Orientation - 'v' for vertical, 'h' for horizontal
        error_bars: Whether to show error bars (standard error)
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure()
    
    # Create plot
    if orient == "h":
        ax = sns.barplot(data=df, y=x_column, x=y_column, hue=hue, palette=palette,
                       errorbar='se' if error_bars else None)
        # Add value annotations
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')
    else:
        ax = sns.barplot(data=df, x=x_column, y=y_column, hue=hue, palette=palette,
                       errorbar='se' if error_bars else None)
        # Add value annotations
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')
    
    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/barplot_{x_column}_{y_column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

@tool
def create_pie_chart(data: List[Dict[str, Any]], column: str, title: str, 
                   explode_largest: bool = True, colors: str = "viridis", 
                   save: bool = True) -> str:
    """
    Create a pie chart showing proportions.
    
    Args:
        data: List of dictionaries containing the data
        column: Column to show distribution for
        title: Title for the plot
        explode_largest: Whether to explode the largest slice
        colors: Color palette
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or description
    """
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Count values in column
    value_counts = df[column].value_counts()
    
    # Create figure
    plt.figure()
    
    # Create explode array if needed
    explode = None
    if explode_largest:
        explode = [0.1 if i == value_counts.argmax() else 0 for i in range(len(value_counts))]
    
    # Create color map
    cmap = plt.cm.get_cmap(colors)
    colors_list = [cmap(i/len(value_counts)) for i in range(len(value_counts))]
    
    # Create plot
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
           startangle=90, explode=explode, colors=colors_list, shadow=True,
           textprops={'fontsize': 12})
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Set title
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PLOTS_FOLDER}/piechart_{column}_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        plt.close()
        return "Plot displayed"

# SQL agent functions
def create_sql_agent():
    """Create the SQL agent for querying the database"""
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        azure_deployment="gpt-4o",
        temperature=0.2,
        max_tokens=16000,
        streaming=True
    )
    
    # Connect to the database with custom dialect options for better performance
    db = SQLDatabase.from_uri(
        DB_URI,
        sample_rows_in_table_info=5,
        view_support=True
    )
    
    # Create SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # Create the SQL agent with advanced prompt
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        system_prompt=SQL_SYSTEM_PROMPT,
        top_k=10
    )
    
    return agent


def create_visualization_agent():
    """Create a visualization agent"""
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        azure_deployment="gpt-4o",
        temperature=0.2,
        max_tokens=16000,
        streaming=True
    )
    
    # Create visualization tools
    tools = [
        execute_sql_query,
        create_count_plot,
        create_distribution_plot,
        create_boxplot,
        create_heatmap,
        create_scatter_plot,
        create_bar_plot,
        create_pie_chart
    ]
    
    # Updated visualization system prompt with explicit instructions
    visualization_system_prompt = """
    You are an advanced HR data visualization specialist. Your job is to create insightful and informative visualizations from HR database query results.
    
    IMPORTANT: When creating visualizations after performing a SQL query:
    1. You MUST convert the DataFrame results to a list of dictionaries using df.to_dict('records')
    2. ALWAYS pass this converted data as the 'data' parameter to ANY visualization function
    3. Do NOT skip this step or the visualization will fail
    
    Example of correct tool usage sequence:
    1. First query: `execute_sql_query` to get data
    2. Store the result: `sql_result = execute_sql_query(...)`
    3. Convert to required format: `data = sql_result.to_dict('records')`
    4. Pass to visualization: `create_scatter_plot(data=data, x_column='...', y_column='...', ...)`
    
    Follow these guidelines:
    1. First, understand what type of visualization would best represent the data.
    2. For categorical data, consider count plots, bar plots, or pie charts.
    3. For numerical data, consider histograms, boxplots, scatter plots, or line plots.
    4. For relationships, use scatter plots, heatmaps, or grouped bar plots.
    5. Always suggest appropriate plot types based on the data structure.
    6. For time series or trend data, use line plots.
    
    The HR database contains employee data with these columns:
    - employee_id: INTEGER (Primary key)
    - first_name: TEXT
    - last_name: TEXT
    - email: TEXT
    - gender: TEXT ('Male' or 'Female')
    - job_title: TEXT
    - department: TEXT (Engineering, Marketing, Sales, HR, Finance)
    - salary: INTEGER
    - hire_date: TEXT (YYYY-MM-DD format)
    - country: TEXT (USA, India, UK, Canada, Germany)
    - city: TEXT (various cities in each country)
    - performance_rating: INTEGER (1-5 scale)
    - years_experience: INTEGER
    
    First, you should decide what data you need from the database and craft an appropriate SQL query.
    Then, choose suitable visualizations to represent the data effectively.
    """
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", visualization_system_prompt),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def create_combined_agent():
    """Create a combined agent that can execute SQL and create visualizations"""
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        azure_deployment="gpt-4o",
        temperature=0.2,
        max_tokens=16000,
        streaming=True
    )
    
    # Create tools
    tools = [
        execute_sql_query,
        create_count_plot,
        create_distribution_plot,
        create_boxplot,
        create_heatmap,
        create_scatter_plot,
        create_bar_plot,
        create_pie_chart
    ]
    
    # Create system prompt combining SQL and visualization WITH explicit instructions for data passing
    combined_system_prompt = """
    You are an advanced HR data analyst with expertise in both SQL querying and data visualization.
    You can query a database and create visualizations based on the results.
    
    First, understand what type of data the user is looking for and write an appropriate SQL query.
    Then, analyze the structure of the returned data and decide on the best visualization(s) to represent it.
    
    IMPORTANT: When using visualization tools after performing a SQL query:
    1. You MUST convert the DataFrame results to a list of dictionaries using df.to_dict('records')
    2. ALWAYS pass this converted data as the 'data' parameter to ANY visualization function
    3. Do NOT skip this step or the visualization will fail
    
    Example of correct tool usage sequence:
    1. First query: `execute_sql_query` to get data
    2. Store the result: `sql_result = execute_sql_query(...)`
    3. Convert to required format: `data = sql_result.to_dict('records')`
    4. Pass to visualization: `create_scatter_plot(data=data, x_column='...', y_column='...', ...)`
    
    The HR database contains an 'employees' table with these columns:
    - employee_id: INTEGER (Primary key)
    - first_name: TEXT
    - last_name: TEXT
    - email: TEXT
    - gender: TEXT ('Male' or 'Female')
    - job_title: TEXT
    - department: TEXT (Engineering, Marketing, Sales, HR, Finance)
    - salary: INTEGER
    - hire_date: TEXT (YYYY-MM-DD format)
    - country: TEXT (USA, India, UK, Canada, Germany)
    - city: TEXT (various cities in each country)
    - performance_rating: INTEGER (1-5 scale)
    - years_experience: INTEGER
    
    Follow these guidelines:
    1. For SQL queries:
       - Use advanced SQL features like CTEs and window functions for complex queries
       - Format SQL queries with proper indentation
       - Use appropriate joins and subqueries
       - Always explain the SQL approach
       
    2. For visualizations:
       - Choose the most appropriate visualization type based on the data structure:
         * Categorical data: bar charts, count plots, or pie charts
         * Numerical data: histograms, boxplots, scatter plots
         * Relationships: scatter plots, heatmaps, bar plots
         * Time series: line plots
       - Always include descriptive titles
       - Add annotations where helpful (like correlation values)
       - Use appropriate color schemes
    
    Always explain your approach and what insights can be gained from the visualization.
    """
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", combined_system_prompt),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def chat_with_hr_analysis_agent():
    """Main function to interact with the combined HR analysis agent"""
    print("\n" + "=" * 80)
    print("🔍 Advanced HR Analysis Agent".center(80))
    print("=" * 80)
    print("\nWelcome to the HR Analysis Assistant!")
    print("This agent can query your database AND create visualizations from the results.")
    print("Type 'exit' to quit, 'sql' for SQL-only mode, 'viz' for visualization-only mode,")
    print("or 'combined' for the integrated SQL + visualization agent,")
    print("Type 'examples' to see some example questions in that mode.")
    
    # Create database indexes for better performance
    create_database_indexes()
    
    # Track the current mode
    current_mode = "combined"
    
    # Initialize agents
    print("\nInitializing agents (this may take a moment)...")
    
    sql_agent = None
    viz_agent = None
    combined_agent = create_combined_agent()
    
    print(f"Agents initialized! Output plots will be saved to the '{PLOTS_FOLDER}' folder.")
    
    while True:
        print("\n" + "-" * 80)
        print(f"Current mode: {current_mode}")
        user_input = input("Your query: ")
        print("-" * 80)
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the HR Analysis Assistant. Goodbye!")
            close_db_connection()
            break
            
        elif user_input.lower() == "sql":
            current_mode = "sql"
            if sql_agent is None:
                print("Initializing SQL agent...")
                sql_agent = create_sql_agent()
            print("Switched to SQL-only mode.")
            continue
            
        elif user_input.lower() == "viz":
            current_mode = "visualization"
            if viz_agent is None:
                print("Initializing visualization agent...")
                viz_agent = create_visualization_agent()
            print("Switched to visualization-only mode.")
            continue
            
        elif user_input.lower() == "combined":
            current_mode = "combined"
            print("Switched to combined SQL + visualization mode.")
            continue
            
        elif user_input.lower() == "examples":
            print_examples(current_mode)
            continue
        
        try:
            print("\nProcessing your request...")
            start_time = time.time()
            
            if current_mode == "sql":
                if sql_agent is None:
                    sql_agent = create_sql_agent()
                response = sql_agent.invoke({"input": user_input})
            elif current_mode == "visualization":
                if viz_agent is None:
                    viz_agent = create_visualization_agent()
                response = viz_agent.invoke({"input": user_input})
            else:  # combined mode
                response = combined_agent.invoke({"input": user_input})
            
            processing_time = time.time() - start_time
            
            print(f"\n✅ Response (generated in {processing_time:.2f} seconds):")
            print(response["output"])
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing your query or check if the database is accessible.")

def print_examples(mode):
    """Print example queries based on the current mode"""
    if mode == "sql":
        print("\nSQL Query Examples:")
        print("=" * 80)
        examples = [
            "What's the average salary for each department?",
            "Who are the top 5 highest paid employees?",
            "Show me the gender distribution in each department.",
            "Calculate the average performance rating by country.",
            "Find employees with above-average salaries within their department.",
            "Which department has the highest average years of experience?",
            "Analyze the correlation between performance and salary using SQL."
        ]
    elif mode == "visualization":
        print("\nVisualization Request Examples:")
        print("=" * 80)
        examples = [
            "Create a count plot of employees by department.",
            "Make a histogram showing the distribution of salaries.",
            "Show me a boxplot of performance ratings by department.",
            "Create a scatter plot of years of experience vs. salary.",
            "Generate a pie chart showing the gender distribution.",
            "Make a bar chart of average salary by country.",
            "Create a heatmap showing correlations between numerical columns."
        ]
    else:  # combined mode
        print("\nCombined Analysis Examples:")
        print("=" * 80)
        examples = [
            "Analyze and visualize the relationship between salary and performance ratings.",
            "Show me a distribution of salaries by department with appropriate visualizations.",
            "Create a visualization showing which departments have the best gender balance.",
            "Analyze the performance ratings across different countries and visualize the results.",
            "Query the database for employees with high experience but low salaries and visualize this distribution.",
            "Create a comprehensive analysis of how performance relates to experience with plots.",
            "Identify and visualize potential salary inequalities between genders across departments."
        ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file {DB_PATH} not found.")
        print("Please ensure the database exists at the specified path.")
        exit(1)
        
    try:
       
        chat_with_hr_analysis_agent()
    finally:
        close_db_connection()











