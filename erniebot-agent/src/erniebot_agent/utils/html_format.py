ITEM_LIST_HTML = """<html>
<head>
    <meta charset="UTF-8">
    <title>Colored List</title>
    <style>
        /* 设置第一个 <li> 元素的颜色为红色 */
        ul.custom-list li:last-child {{
            color: green;
        }}
    </style>
</head>
<body>
    <ul class="custom-list">
        {ITEM}
    </ul>
</body>
</html>"""

IMAGE_HTML = "<img src='data:image/png;base64,{BASE64_ENCODED}' />"
