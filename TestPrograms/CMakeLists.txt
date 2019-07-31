FILE(GLOB FGPU_LIST *.fgpu)

message(${FGPU_LIST})

foreach (FGPU_FILE ${FGPU_LIST})
    get_filename_component(FGPU_FILE_NAME ${FGPU_FILE} NAME)
    configure_file(${FGPU_FILE} ${FGPU_FILE_NAME} COPYONLY)
    message("configure_file(${FGPU_FILE} ${FGPU_FILE_NAME} COPYONLY)")
endforeach()
