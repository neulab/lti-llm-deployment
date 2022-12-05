# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import generation_pb2 as generation__pb2


class GenerationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Generate = channel.unary_unary(
            "/generation.GenerationService/Generate",
            request_serializer=generation__pb2.GenerationRequest.SerializeToString,
            response_deserializer=generation__pb2.GenerationResponse.FromString,
        )


class GenerationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Generate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_GenerationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Generate": grpc.unary_unary_rpc_method_handler(
            servicer.Generate,
            request_deserializer=generation__pb2.GenerationRequest.FromString,
            response_serializer=generation__pb2.GenerationResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("generation.GenerationService", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class GenerationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Generate(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/generation.GenerationService/Generate",
            generation__pb2.GenerationRequest.SerializeToString,
            generation__pb2.GenerationResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
