import os
import re
from typing import Union

import pandas as pd
from vanna import ValidationError
from vanna.exceptions import ImproperlyConfigured, DependencyError

from vanna.ollama import Ollama
from vanna.types import TrainingPlan, TrainingPlanItem

from my_chromadb_vector import My_ChromaDB_VectorStore
# Placeholder for additional imports or code
class MyVanna(My_ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    # Generate follow-up questions
    def generate_followup_questions(
            self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. Respond in Chinese, the user asked the question: '{question}'\n\nThe SQL query for this question is: {sql}\n\nHere is the pandas DataFrame containing the query results: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                f"Respond in Chinese, generate a list of {n_questions} follow-up questions the user might ask based on this data. Reply with a list of questions, one per line. Do not provide any explanations—just the questions. Remember, each question should correspond to a clear SQL query that can be generated from it. Prefer questions that allow for deeper data exploration by slightly modifying the generated SQL query. Each question will become a button that the user can click to generate a new SQL query, so avoid 'example' type questions. Each question must correspond one-to-one with the instantiated SQL query." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    # Generate summary
    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. Respond in Chinese, the user asked the question: '{question}'\n\nHere is the pandas DataFrame containing the query results: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Respond in Chinese, briefly summarize the data based on the question asked. Do not provide any other explanations besides the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)
        return summary

    # Get SQL prompt
    def get_sql_prompt(
            self,
            initial_prompt: str,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ):

        if initial_prompt is None:
            initial_prompt = f"You are an expert in {self.dialect}, respond in English. " + \
                             "Please help generate an SQL query to answer the question. Your response should be based solely on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            " === Response Guidelines === \n "
            "1. If the provided context is sufficient, generate a valid SQL query without explaining the question.\n "
            "2. If the provided context is almost sufficient but requires knowing specific strings in specific columns, generate an intermediate SQL query to find the distinct strings in that column. Add a comment before the query indicating middle_sql.\n "
            "3. If the provided context is insufficient, explain why it is not possible to generate the query.\n "
            "4. Use the most relevant tables.\n "
            "5. If the question has been asked and answered before, repeat the previous answer exactly.\n "
            f"6. Ensure the output SQL conforms to {self.dialect}, is executable, and has no syntax errors.\n "
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))
        return message_log

    # Get follow-up questions prompt
    def get_followup_questions_prompt(
            self,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ) -> list:
        initial_prompt = f"Respond in Spanish, the user's initial question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
            "Respond in English, generate a list of follow-up questions the user might ask based on this data. Reply with a list of questions, one per line. Do not provide any explanations—just the questions."
            )
        )

        return message_log

    # Generate question
    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
            self.system_message(
                "Respond in English, the user will provide you with an SQL query, and you will try to guess what business question this query answers. Return only the question without any additional explanation. Do not reference table names in the question."
            ),
            self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    # Generate plotly code
    def generate_plotly_code(
            self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"Here is a pandas DataFrame containing the results of the query answering the user's question: '{question}'"
        else:
            system_msg = "Here is the pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\n The DataFrame was generated using this query: {sql}\n\n"
            if df_metadata is None:
                raise ValueError("DataFrame metadata is required to generate plotly code.")
        system_msg += f"Here is the information about the generated pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate Python plotly code to plot the results of the DataFrame? Assume the data is in a variable named 'df'. If the DataFrame contains only one value, use an indicator. Reply with only the Python code. Do not provide any explanations—just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # Training
    def train(
            self,
            question: str = None,
            sql: str = None,
            ddl: str = None,
            documentation: str = None,
            plan: TrainingPlan = None,
    ) -> str:
        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(question=question, ddl=ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def connect_to_oracle(
            self,
            user: str = None,
            password: str = None,
            dsn: str = None,
            dbClintPath: str = None,
            **kwargs
    ):

        """
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dns="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:

            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")

        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None
        try:
            oracledb.init_oracle_client(lib_dir=dbClintPath)
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(
                            ';'):  # fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle
