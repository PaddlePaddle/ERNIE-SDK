# File Module

::: erniebot_agent.file
    options:
        summary: true


::: erniebot_agent.file.base
    options:
        summary: true
        members:
        - File


::: erniebot_agent.file.local_file
    options:
        summary: true
        members:
        - LocalFile

::: erniebot_agent.file.remote_file
    options:
        summary: true
        members:
        - RemoteFile

::: erniebot_agent.file.file_manager
    options:
        summary: true
        members:
        - FileManager

::: erniebot_agent.file.global_file_manager_handler
    options:
        summary: true
        members:
        - GlobalFileManagerHandler

::: erniebot_agent.file.remote_file
    options:
        summary: true
        members:
        - AIStudioFileClient