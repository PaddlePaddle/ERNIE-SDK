# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Optional)

__all__ = ['upload_file_to_bos']


def upload_file_to_bos(origin_file: str,
                       upload_file_name: str,
                       category_dir: str='erniebot',
                       bos_host: str='bj.bcebos.com',
                       bos_bucket: str='ernie-bot-sdk',
                       access_key_id: Optional[str]=None,
                       secret_access_key: Optional[str]=None) -> str:
    from baidubce.bce_client_configuration import BceClientConfiguration  # type: ignore
    from baidubce.auth.bce_credentials import BceCredentials  # type: ignore
    from baidubce.services.bos.bos_client import BosClient  # type: ignore
    b_config = BceClientConfiguration(
        credentials=BceCredentials(access_key_id, secret_access_key),
        endpoint=bos_host)
    bos_client = BosClient(b_config)
    with open(origin_file, 'rb') as f:
        bos_client.put_object_from_string(bos_bucket,
                                          f"{category_dir}/{upload_file_name}",
                                          f.read())
    url = f"https://bj.bcebos.com/{bos_bucket}/{category_dir}/{upload_file_name}"
    return url
