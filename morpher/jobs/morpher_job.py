#!/usr/bin/env python
import traceback
import json
import uuid
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from ag_worker.jobs import Job

class MorpherJob(Job):

    def api(self, blueprint, action, data):
        '''
        Exposes API to Worker scripts.
        TODO: How can we secure those endpoints?
        '''
        hostname = self.config.get('morpher', 'hostname')
        port = self.config.get('morpher', 'port')
        
        endpoint = "http://{hostname}:{port}/{blueprint}/{action}/".format(hostname=hostname, port=port, blueprint=blueprint, action=action)
        self.logger.debug("Endpoint: %s" % endpoint)
 
        try:

            request = Request(endpoint, data=json.dumps(data).encode('utf8'), headers={'content-type': 'application/json'})
            response = urlopen(request).read().decode()
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

        


