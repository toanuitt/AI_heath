# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: service.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x12\x0e\x63uff_detection\"R\n\x14\x43uffDetectionRequest\x12\x14\n\x0c\x62\x61se64_image\x18\x01 \x01(\t\x12\x16\n\tthreshold\x18\x02 \x01(\x05H\x00\x88\x01\x01\x42\x0c\n\n_threshold\"Q\n\x15\x43uffDetectionResponse\x12\x10\n\x08position\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x15\n\rerror_message\x18\x03 \x01(\t2}\n\x14\x43uffDetectionService\x12\x65\n\x12\x44\x65tectCuffPosition\x12$.cuff_detection.CuffDetectionRequest\x1a%.cuff_detection.CuffDetectionResponse(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CUFFDETECTIONREQUEST']._serialized_start=33
  _globals['_CUFFDETECTIONREQUEST']._serialized_end=115
  _globals['_CUFFDETECTIONRESPONSE']._serialized_start=117
  _globals['_CUFFDETECTIONRESPONSE']._serialized_end=198
  _globals['_CUFFDETECTIONSERVICE']._serialized_start=200
  _globals['_CUFFDETECTIONSERVICE']._serialized_end=325
# @@protoc_insertion_point(module_scope)
