# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import service_pb2 as service__pb2

GRPC_GENERATED_VERSION = '1.71.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class CuffDetectionServiceStub(object):
    """Service definition with bidirectional streaming
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectCuffPosition = channel.stream_stream(
                '/cuff_detection.CuffDetectionService/DetectCuffPosition',
                request_serializer=service__pb2.CuffDetectionRequest.SerializeToString,
                response_deserializer=service__pb2.CuffDetectionResponse.FromString,
                _registered_method=True)


class CuffDetectionServiceServicer(object):
    """Service definition with bidirectional streaming
    """

    def DetectCuffPosition(self, request_iterator, context):
        """Bidirectional streaming RPC for real-time cuff detection
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CuffDetectionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectCuffPosition': grpc.stream_stream_rpc_method_handler(
                    servicer.DetectCuffPosition,
                    request_deserializer=service__pb2.CuffDetectionRequest.FromString,
                    response_serializer=service__pb2.CuffDetectionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'cuff_detection.CuffDetectionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('cuff_detection.CuffDetectionService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class CuffDetectionService(object):
    """Service definition with bidirectional streaming
    """

    @staticmethod
    def DetectCuffPosition(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/cuff_detection.CuffDetectionService/DetectCuffPosition',
            service__pb2.CuffDetectionRequest.SerializeToString,
            service__pb2.CuffDetectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
