import traceback
import json
import simplejson
import os
import uuid
from urllib.request import Request, urlopen

try:
    from ag_worker.jobs import Job
except ImportError:
    from morpher.jobs import Job

import importlib

class MorpherJob(Job):
    def api(self, blueprint, action, data):
        """
        Exposes API to Worker scripts.
        TODO: How can we secure those endpoints?
        """
        hostname = self.config.get("morpher", "hostname") or self.config.get(
            "hostname"
        )
        port = self.config.get("morpher", "port") or self.config.get("port")
        prefix = self.config.get("morpher", "prefix")
        if prefix:
            endpoint = "http://{hostname}:{port}/{prefix}/{blueprint}/{action}/".format(
                hostname=hostname,
                prefix=prefix,
                port=port,
                blueprint=blueprint,
                action=action,
            )
        else:
            endpoint = "http://{hostname}:{port}/{blueprint}/{action}/".format(
                hostname=hostname,
                port=port,
                blueprint=blueprint,
                action=action,
            )

        self.logger.debug("Endpoint: %s" % endpoint)
        print("Endpoint: %s" % endpoint)

        try:

            request = Request(
                endpoint,
                data=simplejson.dumps(data, ignore_nan=True).encode("utf8"),
                headers={"content-type": "application/json"},
            )
            response = urlopen(request).read().decode()
            return json.loads(response)

        except Exception as e:

            self.logger.error("Could not process request. \n{0}".format(e))
            self.logger.debug("Endpoint: " + endpoint)
            return {"status": "error", "msg": str(e)}

    def save_to_file(self, data, filename=None):
        """
        Stores currently loaded data frame and saves it to a file using a uuid.
        """
        file_path = self.working_dir() + str(uuid.uuid1())
        try:
            # if filename is provided, we also save a persistent copy in the
            # user's directory for ulterior use
            if filename:
                users_path = os.path.abspath(
                    self.config.get("paths", "user_files")
                )
                persistent_file_path = os.path.join(users_path, filename)
                data.to_csv(path_or_buf=persistent_file_path, index=False)
                self.logger.info(
                    "Data stored persistently to file {0}".format(
                        persistent_file_path
                    )
                )

            data.to_csv(path_or_buf=file_path, index=False)
            self.logger.info("Data stored to file {0}".format(file_path))
            return file_path

        except Exception:
            self.logger.error(traceback.format_exc())
            return "error"

    def get_task(self):
        """
        Gets the parameters defined for the given task_id.
        """
        stmt = 'SELECT id, status, pipeline_id, parameters, fastq_readcount, created_at, user FROM worker."TASKS" WHERE id = :task_id;'

        try:
            # return rowproxy as a dict
            task = {
                column: value
                for column, value in self.execute_select_stmt(
                    stmt, {"task_id": self.task_id}
                )
                .fetchone()
                .items()
            }

            if type(task["parameters"]) == str:
                task["parameters"] = json.loads(task["parameters"])

            return task

        except Exception:
            self.logger.error(traceback.format_exc())
            return None

    def get_callable(self, module_name, class_name):
        """
        Gets a callable, i.e., class definition from a pair of module and class
        This is useful to instantiate a class from a given string if needed
        """
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
         



