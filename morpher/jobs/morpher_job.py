#!/usr/bin/env python
import traceback
import json
import uuid
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from ag_worker.jobs import Job
except ImportError:
    print("No AG Worker found or not configured correctly. Using default 'Job' class instead")
    from morpher.jobs import Job

class MorpherJob(Job):

    def api(self, blueprint, action, data):
        '''
        Exposes API to Worker scripts.
        TODO: How can we secure those endpoints?
        '''
        hostname = self.config.get('morpher', 'hostname') or self.config.get('hostname')
        port = self.config.get('morpher', 'port') or self.config.get('port')
        prefix = self.config.get('morpher', 'prefix')
        if prefix:
            endpoint = "http://{hostname}:{port}/{prefix}/{blueprint}/{action}/".format(hostname=hostname, prefix=prefix ,port=port, blueprint=blueprint, action=action)
        else:
            endpoint = "http://{hostname}:{port}/{blueprint}/{action}/".format(hostname=hostname, port=port, blueprint=blueprint, action=action)
        
        self.logger.debug("Endpoint: %s" % endpoint)
        print("Endpoint: %s" % endpoint)
 
        try:

            request = Request(endpoint, data=json.dumps(data).encode('utf8'), headers={'content-type': 'application/json'})
            response = urlopen(request).read().decode()
            print("response: %s" % response)
            return json.loads(response)

        except Exception as e:

            self.logger.error("Could not process request. \n{0}".format(e))
            self.logger.debug("Endpoint: " + endpoint)
            return {"status":"error", "msg":str(e)}

    def save_to_file(self, data):
        '''
        Stores currently loaded data frame and saves it to a file using a uuid.
        '''
        file_path = self.working_dir() + str(uuid.uuid1())
        try:
            data.to_csv(path_or_buf = file_path, index=False)
            self.logger.info("Data stored to file {0}".format(file_path))
            return file_path
        
        except Exception as e:
            self.logger.error(traceback.format_exc())
            return "error"
    
    def get_task(self):
        '''
        Gets the parameters defined for the given task_id.
        '''
        stmt = "SELECT id, status, pipeline_id, parameters, fastq_readcount, created_at, user FROM worker.\"TASKS\" WHERE id = :task_id";
        
        try:
            #return rowproxy as a dict
            task = {column: value for column, value in self.execute_select_stmt(stmt, {"task_id" : self.task_id}).fetchone().items()}
            return task
        
        except Exception as e:
            self.logger.error(traceback.format_exc())
            return None

        


