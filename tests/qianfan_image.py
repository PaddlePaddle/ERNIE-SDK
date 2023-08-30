import requests
import json
import time

API_KEY = "xyyIHrS3mO7CfMP0421tNLH0"
SECRET_KEY = "a3cI8CZcXbyRTOz5QbFi2oooU92cNL4n"
ACCESS_TOKEN = "25.eb2d147873f3b377991a233d36885d38.315360000.2007532950.282335-37727334"


def gen_image_v1():

    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2img?access_token=" + get_access_token(
    )

    payload = json.dumps({
        "text": "画一个驴肉火烧",
        "resolution": "1024*1024",
        "style": "写实风格"
    })
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    res = json.loads(response.text)
    return res['data']['taskId']


def get_image_v1(task_id):
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImg?access_token=" + get_access_token(
    )
    payload = json.dumps({"taskId": str(task_id), })
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = None
    for _ in range(10):
        response = requests.request("POST", url, headers=headers, data=payload)
        res = json.loads(response.text)
        status = res['data']['status']
        print("status=", res['data']['taskId'])
        if status == 1:
            print("SUCCESS!\n", res)
            break
        else:
            print("getImg status={}, sleep then retry".format(status))
            time.sleep(2)


def gen_image_v2():
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2imgv2?access_token=" + get_access_token(
    )

    payload = json.dumps({
        "prompt": "画一个胸有成竹的男人",
        "version": "v2",
        "width": 1024,
        "height": 1024,
        "image_num": 1
    })
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    res = json.loads(response.text)
    return res['data']['task_id']


def get_image_v2(task_id):
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImgv2?access_token=" + get_access_token(
    )
    payload = json.dumps({"task_id": str(task_id), })
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    response = None
    for _ in range(10):
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        res = json.loads(response.text)
        status = res['data']['task_status']
        if status == "SUCCESS":
            print("SUCCESS!\n", res)
            break
        else:
            print("getImg status={}, sleep then retry".format(status))
            time.sleep(2)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
    }
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    print("Get ERNIE-ViLG v1 API")
    task_id = gen_image_v1()
    print("ERNIE-ViLG v1 task_id=", task_id)
    get_image_v1(task_id)

    print("Get ERNIE-ViLG v2 API")
    task_id = gen_image_v2()
    print("ERNIE-ViLG v2 task_id=", task_id)
    get_image_v2(task_id)
