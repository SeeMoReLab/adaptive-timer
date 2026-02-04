import gbft_pb2_grpc
import gbft_pb2
import logging
import numpy as np
import pandas as pd
import sklearn 
import argparse
import grpc
import queue
import random
from datetime import datetime
from concurrent import futures
from threading import Condition, Lock
from constants import *

proc_queue = queue.Queue()
proc_queue.put("pbft") #Assume first protocol is pbft
CURR_PROC = "pbft"

class EntityCommServicer(gbft_pb2_grpc.EntityCommServicer):

    def send_decision(self, request, context):
        global CURR_PROC
        proc_queue.put(request.next_protocol)

        if request.next_protocol != 'repeat':
            CURR_PROC = request.next_protocol

        return gbft_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

parser = argparse.ArgumentParser(description='Start a client.')
parser.add_argument('--unit', '-u', type=int, required=True, help='Unit id of the corresponding bedrock entity')
parser.add_argument('--port', '-p', type=int, required=True, help='Port of the corresponding bedrock entity')
parser.add_argument('--episodes', '-e', type=int, default=10, help="Number of episodes to be run [Optional]")

args = parser.parse_args()

def send_data(stub):
    for e in range(args.episodes):   
        data = gbft_pb2.LearningData(
            next_protocol= CURR_PROC
        )
        data.report[FAST_PATH_FREQUENCY] = float(random.randint(0, 100))
        data.report[SLOWNESS_OF_PROPOSAL] = float(random.randint(0, 100))
        data.report[REQUEST_SIZE] = float(random.randint(0, 100))
        data.report[RECEIVED_MESSAGE_PER_SLOT] = float(random.randint(0, 1))
        data.report[HAS_FAST_PATH] = float(random.randint(0, 1))
        data.report[HAS_LEADER_ROTATION] = float(random.randint(0, 1))

        try:
            response = stub.send_data(data) 
            logging.info(f"Data sent successfully! Server responded: {response}")
        except grpc.RpcError as e:
            logging.info(f"Failed to send data: {e.details()}")

        request = proc_queue.get()
        logging.info(f"Received decision: {request}")



if __name__ == '__main__':
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

    random.seed(0)
    np.random.seed(0)

    bedrockRPCPort = args.port + 10
    agentPort = args.port + 20
    client_address = '[::]:{}'.format(bedrockRPCPort)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gbft_pb2_grpc.add_EntityCommServicer_to_server(EntityCommServicer(), server)
    server.add_insecure_port(client_address)
    server.start()
    logging.info('grpc client running at %s.', client_address)
    
    try:
        entity_channel = grpc.insecure_channel('localhost:{}'.format(agentPort))
        agent_stub = gbft_pb2_grpc.AgentCommStub(entity_channel)
        send_data(agent_stub)
        server.wait_for_termination()

    finally:
        entity_channel.close()