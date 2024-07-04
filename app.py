""" Main application file for the Flask web server. """
import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from sqlalchemy import create_engine
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_VERSION"]="2023-07-01-preview"
os.environ["OPENAI_CHAT_MODEL"]="gpt-35-turbo"
os.environ["SQL_SERVER"]="joi.database.windows.net;PORT=1433"
os.environ["SQL_DB"]="northwind"

DRIVER = '{ODBC Driver 17 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+DRIVER+ \
                ';Server=tcp:' + os.getenv("SQL_SERVER")+ \
                ';DATABASE=' + os.getenv("SQL_DB") + \
                ';Uid=' + os.getenv("SQL_USERNAME")+ \
                ';Pwd=' + os.getenv("SQL_PWD") + \
                ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

db_engine = create_engine(odbc_str)

llm = AzureChatOpenAI(
    model_name=os.getenv("OPENAI_CHAT_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
# pylint: disable=line-too-long
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are an agent designed to interact with a SQL database Northwind.
         Given an input question, create a syntactically correct {dialect} query to run, 
         then look at the results of the query and return the answer. 
         Unless the user specifies a specific number of examples they wish to obtain, 
         always limit your query to at most 10 results.
         You can order the results by a relevant column to return the most interesting examples in the database.
         Never query for all the columns from a specific table, only ask for the relevant columns given the question.
         You have access to tools for interacting with the database.
         Only use the given tools. Only use the information returned by the tools to construct your final answer.
         You MUST double check your query before executing it. If you get an error while executing a query, 
         rewrite the query and try again.
         
         DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
         If the question does not seem related to the database, just return "No lo sé, pero estoy aprendiendo" as the answer.
         
         Answer in Spanish.
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)
# pylint: enable=line-too-long

db = SQLDatabase(db_engine)

sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

sqldb_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    #agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

app = Flask(__name__)


@app.route('/')
def index():
    """ Main route for the web server. """    
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """ Route for the favicon. """
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
    """ Route for the hello page."""    
    name = request.form.get('name')
    try:
        resultado = sqldb_agent.invoke(final_prompt.format(
            question=name,
            dialect="SQL Server"
            ))
    except Exception as e:
        print(e)
        resultado = "No lo sé, pero estoy aprendiendo"

    # resultado es un json que tiene input y output. output es el resultado de la query
    resultado = resultado['output']

    if name:
        print('Request for hello page received with name=%s', name)
        print('Resultado: ', resultado)
        return render_template('hello.html', name = resultado)
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run()
