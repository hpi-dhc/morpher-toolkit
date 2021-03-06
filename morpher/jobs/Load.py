import traceback
from morpher.exceptions import kwargs_not_empty
import morpher.config as config
import pandas as pd
import pyhdb
from morpher.jobs import MorpherJob


class Load(MorpherJob):
    def do_execute(self):

        filename = self.get_input("filename")
        task = self.get_task()

        data = self.execute(filename=filename)
        self.add_output("filename", filename)
        self.add_output("cohort_id", task["parameters"]["cohort_id"])
        self.add_output("user_id", task["parameters"]["user_id"])

        return data

    def execute(self, source=config.FILE, **kwargs):

        data = pd.DataFrame()

        try:
            if source == config.FILE:
                kwargs_not_empty(kwargs.get("filename"), "filename")
                data = self.load_from_file(kwargs.get("filename"))

            elif source == config.DATABASE:
                kwargs_not_empty(kwargs.get("sql"), "sql")
                data = self.load_from_server(kwargs.get("sql"))

        except Exception:
            self.logger.error(traceback.format_exc())

        return data

    def load_from_server(self, sql):
        """
        Opens a database connection to the HANA backend.
        """
        try:
            connection = pyhdb.connect(
                host=config.db_address,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
            )
            with open(sql, "r") as f:
                query = f.read()
            cursor = connection.cursor()
            cursor = cursor.execute(query)
            data = pd.DataFrame(cursor.fetchall())
        except Exception as e:
            print(
                "***Error connecting to DB. Check your environment variables.\n",
                repr(e),
            )
            raise e
        return data

    @staticmethod
    def load_from_file(filename):
        """
        Reads data from file.

        """
        data = pd.read_csv(filepath_or_buffer=filename)
        return data
