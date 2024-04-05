# nms

```bash
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

       Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]

    步骤一：将所有矩形框按照不同的类别标签分组，组内按照置信度高低得分进行排序；
    步骤二: 将步骤一中得分最高的矩形框拿出来,遍历剩余矩形框,计算与当前得分最高的矩形框的交并比,将剩余矩形框中大于设定的IOU阈值的框删除
    步骤三：将步骤二结果中，对剩余的矩形框重复步骤二操作，直到处理完所有矩形框；

    """

    nc = prediction.shape[2] - 5  # number of classes   prediction是网络模型的直接输出, 输出其shape是(1, 50000（以50000为例子.这个是网络的候选框）, nc+5), 1表示的是图片的个数，50000表示是网络预测的候选框的个数，nc + 5 表示xywh/4 含有目标的概率/1，每个nc的概率/80（4+1+80）
    xc = prediction[..., 4] > conf_thres  # candidates  prediction[…, 4]是shape 为torch.Size([1, 50000]) tensor，prediction[…, 4]意义是取所有预测值的第5个值，表示目标框含有目标的概率值，整个表达式prediction[…, 4] > conf_thres，返回的是一个与prediction[…, 4]具有相同shape（1, 50000）的tensor，其每个值是True或者False，然后把这个tensor赋值给xc，所以xc的shape是（1, 50000），每个值是True或者False，用其值表达每个box的置信度是否大于或者小于conf_thres值。
    # ... 表示选择所有维度的数据
    # 这个判断的是第五列（obj_conf）是否大于conf_thres,之所以可以这么判断，是因为这些都是小于1的数字，所以小于conf_thres的数，他的conf一定会小于conf_thres的
    # 这个是计算conf的公式，计算的代码大概在738行左右，conf = obj_conf * cls_conf
	
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height （像素值）最小和最大的box的宽和高
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms() 输送到torchvision.ops.nms()接口中最大box总数量
    time_limit = 10.0  # seconds to quit after  nms函数执行超时设置
    redundant = True  # require redundant detections 需要额外的检测
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0] # output被定义成一个list，list的长度等于预测图片的个数prediction.shape[0]，每个list的元素是一个包含6个字段的tensor。
    for xi, x in enumerate(prediction):  # image index, image inference 这个xi获取了第一张图片所有可能是正样本的位置，【相当于降维】
        # Apply constraints
        
        # xi是图片的index，其值是0，表示第0张图片，x是图片的推理结果，其shape是torch.Size([50000, 5+nc])，这里巧妙的利用了enumerate将图片的index和图片的推理结果分离开，分别存放在了xi和x里面。本来prediction是一个shape为torch.Size([1, 50000, 7])的三维张量，张量的轴1是图片的index，张量的另外两个轴对应图片的推理结果，这里利用了enumerate将prediction的这两部分给分离开了。这行代码很巧妙
        
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height 将数组x中特定范围之外的值所在行的第5列（索引为4）的元素设为0。这个是把obj设为0了（舍弃该目标），前4列是xywh，第五列是obj，后面80列是每一类的概率
        x = x[xc[xi]]  # confidence
        
        """
        xc是之前计算出的shape为(1, 50000)的tensor, 每个值是True或者False, 【大于置信度的阈值为True】, 用其值表达每个box的置信度是否大于或者小于conf_thres值。xc[xi], [xi]这个是对xc tensor的取值方式, xi的值是0, 则表示取第0张图片的所有True或者False值【看xc的维度,准确的说应该是一个三维的tensor】。xc[xi]的shape是torch.Size([50000]), 与x的第0个维度一致, x[xc[xi]]也是对x tensor的取值方式, x[xc[xi]]整个表达式则表示对x第0个维度上xc[xi]所有为True的保留, 为False的则舍弃, 也就是对第i(这里i是0)张图片所有confidence值大于conf_thres的box取出, 然后重新赋值给x。
        """

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]: # 检测该图片是否有大于conf_thres的box，有的进一步处理，没有的话进入下一次循环
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # 这行是计算每个框所预测的类别的概率，yolo论文的公式是 Pr(Classi | Object) * Pr(Object) * IoU(tuth，pred) = Pr(classi) * IoU(tuth，pred)
        # 这个就是，有目标的概率 * 该目标是该类别的概率 = conf
        """
        这个引自于yolo论文gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the
        object. 从这句话可以看出来,这个公式计算得到的scores值将每个类别的概率值以及预测框对目标框定范围的准确程度,而代码没有使用IoU这一部分,仅仅只做了类别的概率计算。x[:, 5:]对应的是 Pr(Classi | Object) ,x[:, 4:5]对应的是Pr(Object)。而概率的计算也很巧妙,x[:, 5:]对应5000个nc列【2列,以两列为例子】tensor,x[:, 4:5],对应5000个1列tensor,x[:, 5:] *= x[:, 4:5]则表示为nc列【2列】的每个元素和一列的每个元素分别相乘得到nc列【2列】的元素, 元素再赋值给x[:, 5:]的nc列【2列】。经过这一步计算x的后面nc列【两列(第6列,第7列)】的值就是代表了目标的所在类别的confidence值了。
        """

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # 原来yolo的输出box的格式是center x, center y, width, height，经过这一步后将box的四个值表达为(x1, y1, x2, y2)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            """
            这里使用了tensor的max函数,这个函数的详细用法可以参考https://www.jianshu.com/p/3ed11362b54f, max的输入参数是1表示对每行求最大值,函数会返回两个tensor,第一个tensor是每行的最大值, 第二个tensor是每行最大值的索引。keepdim可以参考https://blog.csdn.net/zylooooooooong/article/details/112576268, 表示输出维度和输入维度是否一致,True则表示一致。函数返回的conf的shape是torch.Size([50000, 1]),每一行是x[:, 5:]的最大值,j是每一行最大值的索引,这个索引最终用来表示成nms输出的每个类别的id。
            """
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            """
            这一行将box,conf,以及j 按照列cat成一个tensor作为网络的输出。[conf.view(-1) > conf_thres]则是筛选出出confidence值大于conf_thres所有box。
            代码执行到这一行,基本就算是把置信度大于conf_thres的所有box给筛选出来了,筛选结果存放在x tensor里面。x的0到3列存放box,4列存放conf,5列存放class种类id。
            """

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)] # 这一行是利用class进行过滤，筛选出指定的class，nms仅仅对指定的class进行nms。

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence 如果box的个数超过最大nms个数则按照confidence值降序排列，取出置信度最大的前max_nms的box做nms

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes  agnostic 类别无关nms，不只是同类别进行nms
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        """
        c = x[:, 5:6] * (0 if agnostic else max_wh)这行代码是多类别中应用NMS具体意义可以参考https://blog.csdn.net/flyfish1986/article/details/119177472。多类别NMS(非极大值抑制)的处理策略是为了让每个类都能独立执行NMS,在所有的边框上添加一个偏移量。偏移量仅取决于类的ID(也就是x[:, 5:6]),并且足够大,以便来自不同类的框不会重叠。
        boxes, scores = x[:, :4] + c, x[:, 4] # boxes (offset by class), scores
        这行取出boxes和scores,boxes添加了偏移量c,不通过类别的偏移量大小不一致。
        i = torchvision.ops.nms(boxes, scores, iou_thres) # NMS
        调用torch自带的nms接口实现重叠框的抑制,函数返回的是一个tensor i
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        i的意义是,整型64张量,指示被保留的框的index,另外是按照得分（置信度）从高到低排列

        """
        if i.shape[0] > max_det:  # limit detections 判断是否超过最大nms检测个数，如果超过，则去掉置信度低的。
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i] # x[i]是利用nms的结果i,取出所有nms 结果i对应的box，然后将结果保存到xi张图片对应的output里面。
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
```

**xc = prediction[..., 4] > conf_thres：详细操作**

![image-20240404104107773](.\assert\image-20240404104107773.png)

**x = x[xc[xi]]  例子**

![Snipaste_2024-04-04_16-34-31](.\assert\Snipaste_2024-04-04_16-34-31.png)

**conf, j = x[:, 5:].max(1, keepdim=True)** 545的索引是它所在行的第1【索引】个数字，因此输出的是1，但是放在第二行了，keepdim = True【保持维度不变】，这样就能找到这个置信度对应的class

![Snipaste_2024-04-04_17-01-27](.\assert\Snipaste_2024-04-04_17-01-27.png)

**x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] ** ~~第二次过滤小于阈值的框~~，但是还没有过滤掉IoU较大的框

![Snipaste_2024-04-04_17-27-13](.\assert\Snipaste_2024-04-04_17-27-13.png)

排序后面会调用 i = torchvision.ops.nms(boxes, scores, iou_thres)进行排序，这个i返回的是经过IoU处理且置信度从大到小排序的结果
