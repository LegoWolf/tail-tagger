import logging
import sys
import time
import win32serviceutil
import win32service
import servicemanager

import e621_classifier

class ClassifierService:
    def stop(self):
        """Stop the service""" 
        logging.info("Service stopping...")
        self.running = False

    def run(self):
        """Main service loop. This is where work is done!"""
        self.running = True

        app = e621_classifier.Application()
        if app.start():
            logging.info("Service started.")
            try:
                while self.running:
                    time.sleep(1)

            except Exception as e:
                logging.error(e)

            finally:
                app.stop()

class ClassifierServiceFramework(win32serviceutil.ServiceFramework):
    _svc_name_ = 'e621Classifier'
    _svc_display_name_ = 'e621 Classifier'
    _svc_description_ = 'Classifies images in specified folders using e621 tags'

    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.service_impl.stop()
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        """Start the service; does not return until stopped"""
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        self.service_impl = ClassifierService()
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        # Run the service
        self.service_impl.run()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(ClassifierServiceFramework)
        servicemanager.StartServiceCtrlDispatcher()
    else:
         win32serviceutil.HandleCommandLine(ClassifierServiceFramework)
