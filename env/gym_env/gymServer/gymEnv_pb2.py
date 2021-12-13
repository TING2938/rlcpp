# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gymEnv.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gymEnv.proto',
  package='gymEnv',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0cgymEnv.proto\x12\x06gymEnv\"O\n\x05Space\x12\t\n\x01n\x18\x01 \x01(\x05\x12\r\n\x05shape\x18\x02 \x03(\x05\x12\x0c\n\x04high\x18\x03 \x03(\x02\x12\x0b\n\x03low\x18\x04 \x03(\x02\x12\x11\n\tbDiscrete\x18\x05 \x01(\x08\"Q\n\x08\x45nvSpace\x12#\n\x0c\x61\x63tion_space\x18\x01 \x01(\x0b\x32\r.gymEnv.Space\x12 \n\tobs_space\x18\x02 \x01(\x0b\x32\r.gymEnv.Space\"\x1a\n\x0bObservation\x12\x0b\n\x03obs\x18\x01 \x03(\x02\"\x18\n\x06\x41\x63tion\x12\x0e\n\x06\x61\x63tion\x18\x01 \x03(\x02\"Q\n\nStepResult\x12%\n\x08next_obs\x18\x01 \x01(\x0b\x32\x13.gymEnv.Observation\x12\x0e\n\x06reward\x18\x02 \x01(\x02\x12\x0c\n\x04\x64one\x18\x03 \x01(\x08\"\x12\n\x03Msg\x12\x0b\n\x03msg\x18\x01 \x01(\t2\xdb\x01\n\nGymService\x12\'\n\x04make\x12\x0b.gymEnv.Msg\x1a\x10.gymEnv.EnvSpace\"\x00\x12+\n\x05reset\x12\x0b.gymEnv.Msg\x1a\x13.gymEnv.Observation\"\x00\x12,\n\x04step\x12\x0e.gymEnv.Action\x1a\x12.gymEnv.StepResult\"\x00\x12$\n\x06render\x12\x0b.gymEnv.Msg\x1a\x0b.gymEnv.Msg\"\x00\x12#\n\x05\x63lose\x12\x0b.gymEnv.Msg\x1a\x0b.gymEnv.Msg\"\x00\x62\x06proto3'
)




_SPACE = _descriptor.Descriptor(
  name='Space',
  full_name='gymEnv.Space',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='n', full_name='gymEnv.Space.n', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shape', full_name='gymEnv.Space.shape', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='high', full_name='gymEnv.Space.high', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='low', full_name='gymEnv.Space.low', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bDiscrete', full_name='gymEnv.Space.bDiscrete', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=103,
)


_ENVSPACE = _descriptor.Descriptor(
  name='EnvSpace',
  full_name='gymEnv.EnvSpace',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='action_space', full_name='gymEnv.EnvSpace.action_space', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='obs_space', full_name='gymEnv.EnvSpace.obs_space', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=105,
  serialized_end=186,
)


_OBSERVATION = _descriptor.Descriptor(
  name='Observation',
  full_name='gymEnv.Observation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obs', full_name='gymEnv.Observation.obs', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=188,
  serialized_end=214,
)


_ACTION = _descriptor.Descriptor(
  name='Action',
  full_name='gymEnv.Action',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='gymEnv.Action.action', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=216,
  serialized_end=240,
)


_STEPRESULT = _descriptor.Descriptor(
  name='StepResult',
  full_name='gymEnv.StepResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='next_obs', full_name='gymEnv.StepResult.next_obs', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reward', full_name='gymEnv.StepResult.reward', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='done', full_name='gymEnv.StepResult.done', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=242,
  serialized_end=323,
)


_MSG = _descriptor.Descriptor(
  name='Msg',
  full_name='gymEnv.Msg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='gymEnv.Msg.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=325,
  serialized_end=343,
)

_ENVSPACE.fields_by_name['action_space'].message_type = _SPACE
_ENVSPACE.fields_by_name['obs_space'].message_type = _SPACE
_STEPRESULT.fields_by_name['next_obs'].message_type = _OBSERVATION
DESCRIPTOR.message_types_by_name['Space'] = _SPACE
DESCRIPTOR.message_types_by_name['EnvSpace'] = _ENVSPACE
DESCRIPTOR.message_types_by_name['Observation'] = _OBSERVATION
DESCRIPTOR.message_types_by_name['Action'] = _ACTION
DESCRIPTOR.message_types_by_name['StepResult'] = _STEPRESULT
DESCRIPTOR.message_types_by_name['Msg'] = _MSG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Space = _reflection.GeneratedProtocolMessageType('Space', (_message.Message,), {
  'DESCRIPTOR' : _SPACE,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.Space)
  })
_sym_db.RegisterMessage(Space)

EnvSpace = _reflection.GeneratedProtocolMessageType('EnvSpace', (_message.Message,), {
  'DESCRIPTOR' : _ENVSPACE,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.EnvSpace)
  })
_sym_db.RegisterMessage(EnvSpace)

Observation = _reflection.GeneratedProtocolMessageType('Observation', (_message.Message,), {
  'DESCRIPTOR' : _OBSERVATION,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.Observation)
  })
_sym_db.RegisterMessage(Observation)

Action = _reflection.GeneratedProtocolMessageType('Action', (_message.Message,), {
  'DESCRIPTOR' : _ACTION,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.Action)
  })
_sym_db.RegisterMessage(Action)

StepResult = _reflection.GeneratedProtocolMessageType('StepResult', (_message.Message,), {
  'DESCRIPTOR' : _STEPRESULT,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.StepResult)
  })
_sym_db.RegisterMessage(StepResult)

Msg = _reflection.GeneratedProtocolMessageType('Msg', (_message.Message,), {
  'DESCRIPTOR' : _MSG,
  '__module__' : 'gymEnv_pb2'
  # @@protoc_insertion_point(class_scope:gymEnv.Msg)
  })
_sym_db.RegisterMessage(Msg)



_GYMSERVICE = _descriptor.ServiceDescriptor(
  name='GymService',
  full_name='gymEnv.GymService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=346,
  serialized_end=565,
  methods=[
  _descriptor.MethodDescriptor(
    name='make',
    full_name='gymEnv.GymService.make',
    index=0,
    containing_service=None,
    input_type=_MSG,
    output_type=_ENVSPACE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='reset',
    full_name='gymEnv.GymService.reset',
    index=1,
    containing_service=None,
    input_type=_MSG,
    output_type=_OBSERVATION,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='step',
    full_name='gymEnv.GymService.step',
    index=2,
    containing_service=None,
    input_type=_ACTION,
    output_type=_STEPRESULT,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='render',
    full_name='gymEnv.GymService.render',
    index=3,
    containing_service=None,
    input_type=_MSG,
    output_type=_MSG,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='close',
    full_name='gymEnv.GymService.close',
    index=4,
    containing_service=None,
    input_type=_MSG,
    output_type=_MSG,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_GYMSERVICE)

DESCRIPTOR.services_by_name['GymService'] = _GYMSERVICE

# @@protoc_insertion_point(module_scope)
