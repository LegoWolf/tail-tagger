import argparse
import logging
import time
import e621_classifier

if __name__ == '__main__':
    app = e621_classifier.Application()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help=f"path to the {app.get_config_filename()} file.")
    parser.add_argument(
        "--loglevel",
        choices=["debug", "info", "warning", "error"],
        help="minimum level of messages to log")
    args = parser.parse_args()

    if args.config:
        app.set_config_path(args.config)

    if args.loglevel:
        app.set_log_level(args.loglevel)

    if app.start():
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Process aborted at keyboard!")

        except Exception as e:
            logging.error(e)

        finally:
            app.stop()
