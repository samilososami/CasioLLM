# Build with: fxsdk build-cg
cmake_minimum_required(VERSION 3.15)
project(LLM_Casio_FX50 C)

include(GenerateG3A)
find_package(Gint 2.9 REQUIRED)

set(CASIOLLM_BACKEND "tinytalk" CACHE STRING
  "Runtime to build: tinytalk, nanolm-legacy, or nanolm-extreme")
set_property(CACHE CASIOLLM_BACKEND PROPERTY STRINGS
  tinytalk nanolm-legacy nanolm-extreme)
set(CASIOLLM_NANOLM_PROMPT "identity" CACHE STRING
  "NanoLM fixed prompt: identity or short")
set_property(CACHE CASIOLLM_NANOLM_PROMPT PROPERTY STRINGS identity short)

if(CASIOLLM_BACKEND STREQUAL "tinytalk")
  set(CASIOLLM_RUNTIME src/tinytalk.c)
  set(CASIOLLM_OUTPUT "CasioLLM.g3a")
  set(CASIOLLM_VERSION "0.8.1")
  add_compile_definitions(
    CASIOLLM_TINYTALK_COALESCE_STEPS=32
    IO_BUFFER_BYTES=49152)
elseif(CASIOLLM_BACKEND STREQUAL "nanolm-legacy")
  set(CASIOLLM_RUNTIME src/nanolm.c)
  set(CASIOLLM_OUTPUT "CasioLLM-NanoLegacy.g3a")
  set(CASIOLLM_VERSION "0.3.0")
  add_compile_definitions(CASIOLLM_BACKEND_NANOLM=1)
elseif(CASIOLLM_BACKEND STREQUAL "nanolm-extreme")
  set(CASIOLLM_RUNTIME src/nanolm.c)
  if(CASIOLLM_NANOLM_PROMPT STREQUAL "short")
    set(CASIOLLM_OUTPUT "CasioLLM-NanoExtremeShort.g3a")
    set(CASIOLLM_VERSION "0.5.1")
    add_compile_definitions(
      CASIOLLM_NANOLM_SHORT_PROMPT=1
      CASIOLLM_NANOLM_SHORT_IDENTITY=1)
  elseif(CASIOLLM_NANOLM_PROMPT STREQUAL "identity")
    set(CASIOLLM_OUTPUT "CasioLLM-NanoExtreme.g3a")
    set(CASIOLLM_VERSION "0.5.0")
  else()
    message(FATAL_ERROR
      "Unknown CASIOLLM_NANOLM_PROMPT=${CASIOLLM_NANOLM_PROMPT}")
  endif()
  add_compile_definitions(
    CASIOLLM_BACKEND_NANOLM=1
    CASIOLLM_NANOLM_EXTREME=1
    CASIOLLM_COALESCE_STEPS=8)
else()
  message(FATAL_ERROR "Unknown CASIOLLM_BACKEND=${CASIOLLM_BACKEND}")
endif()

add_executable(llm src/main.c ${CASIOLLM_RUNTIME})
target_compile_options(llm PRIVATE -Wall -Wextra -Werror -Os -g -flto)
target_link_libraries(llm Gint::Gint m)

# The visible title is intentionally short: the CG-50 menu only allows 8 bytes.
generate_g3a(TARGET llm OUTPUT "${CASIOLLM_OUTPUT}" NAME "CasioLLM" VERSION "${CASIOLLM_VERSION}"
  ICONS assets-cg/icon-uns.png assets-cg/icon-sel.png)
