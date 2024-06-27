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

connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]

odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                + connection_string

db_engine = create_engine(odbc_str)

llm = AzureChatOpenAI(
    model_name=os.getenv("OPENAI_CHAT_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are an agent designed to interact with a SQL database.
         Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
         Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
         You can order the results by a relevant column to return the most interesting examples in the database.
         Never query for all the columns from a specific table, only ask for the relevant columns given the question.
         You have access to tools for interacting with the database.
         Only use the given tools. Only use the information returned by the tools to construct your final answer.
         You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
         
         DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
         If the question does not seem related to the database, just return "No lo sé, pero estoy aprendiendo" as the answer.
         
         Answer in Spanish.
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)

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
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')
   resultado = sqldb_agent.invoke(final_prompt.format(
       question=name,
       dialect="SQL Server"
   ))
   # resultado es un json que tiene input y output. output es el resultado de la query
   resultado = resultado['output']

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = resultado)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()
