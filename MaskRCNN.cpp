#include "MaskRCNN.h"
#include "Graphics/CGraphicsLayer.h"

CMaskRCNN::CMaskRCNN() : COcvDnnProcess()
{
    m_pParam = std::make_shared<CMaskRCNNParam>();
    setOutputDataType(IODataType::IMAGE_LABEL, 0);
    addOutput(std::make_shared<CImageIO>());
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

CMaskRCNN::CMaskRCNN(const std::string& name, const std::shared_ptr<CMaskRCNNParam> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CMaskRCNNParam>(*pParam);
    setOutputDataType(IODataType::IMAGE_LABEL, 0);
    addOutput(std::make_shared<CImageIO>());
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

size_t CMaskRCNN::getProgressSteps()
{
    return 3;
}

int CMaskRCNN::getNetworkInputSize() const
{
    int size = 800;

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
        size = size + (m_sign * 32);

    return size;
}

double CMaskRCNN::getNetworkInputScaleFactor() const
{
    return 1.0;
}

cv::Scalar CMaskRCNN::getNetworkInputMean() const
{
    return cv::Scalar();
}

std::vector<cv::String> CMaskRCNN::getOutputsNames() const
{
    auto outNames = COcvDnnProcess::getOutputsNames();
    outNames.push_back("detection_out_final");
    return outNames;
}

void CMaskRCNN::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    //Force model files path
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString();
    pParam->m_structureFile = pluginDir + "/Model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    pParam->m_modelFile = pluginDir + "/Model/frozen_inference_graph.pb";
    pParam->m_labelsFile = pluginDir + "/Model/coco_names.txt";

    CMat imgSrc = pInput->getImage();
    std::vector<cv::Mat> netOutputs;
    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;

            if(m_classNames.empty())
                readClassNames();

            generateColors();
        }

        int size = getNetworkInputSize();
        double scaleFactor = getNetworkInputScaleFactor();
        cv::Scalar mean = getNetworkInputMean();
        auto inputBlob = cv::dnn::blobFromImage(imgSrc, scaleFactor, cv::Size(size,size), mean, false, false);

        m_net.setInput(inputBlob);

        Utils::CTimer inferenceTime;
        inferenceTime.start();
        auto netOutNames = getOutputsNames();
        m_net.forward(netOutputs, netOutNames);
        auto t = inferenceTime.get_elapsed_ms();
        m_customInfo.clear();
        m_customInfo.push_back(std::make_pair("Inference time (ms)", std::to_string(t)));
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    emit m_signalHandler->doProgress();

    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();

    endTaskRun();

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
    {
        m_sign *= -1;
        m_bNewInput = false;
    }
}

void CMaskRCNN::manageOutput(std::vector<cv::Mat> &netOutputs)
{
    forwardInputImage(0, 1);
    manageMaskRCNNOutput(netOutputs);   
    //generateColorMap(netOutputs[0], false);
    setOutputColorMap(1, 0, m_colors);
}

void CMaskRCNN::manageMaskRCNNOutput(std::vector<cv::Mat> &netOutputs)
{
    auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));

    //Graphics output
    auto pGraphicsOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(2));
    pGraphicsOutput->setNewLayer(getName());
    pGraphicsOutput->setImageIndex(1);

    //Measures output
    auto pMeasureOutput = std::dynamic_pointer_cast<CBlobMeasureIO>(getOutput(3));
    pMeasureOutput->clearData();

    CMat imgSrc = pInput->getImage();
    cv::Mat labelImg = cv::Mat::zeros(imgSrc.rows, imgSrc.cols, CV_8UC1);
    int nbDetections = netOutputs[1].size[2];

    for(int n=0; n<nbDetections; ++n)
    {
        //Detected class
        int classIndex[4] = { 0, 0, n, 1 };
        size_t classId = (size_t)netOutputs[1].at<float>(classIndex);
        //Confidence
        int confidenceIndex[4] = { 0, 0, n, 2 };
        float confidence = netOutputs[1].at<float>(confidenceIndex);

        if(confidence > pParam->m_confidence)
        {
            //Bounding box
            int leftIndex[4] = { 0, 0, n, 3 };
            int topIndex[4] = { 0, 0, n, 4 };
            int rightIndex[4] = { 0, 0, n, 5 };
            int bottomIndex[4] = { 0, 0, n, 6 };
            float left = netOutputs[1].at<float>(leftIndex) * imgSrc.cols;
            float top = netOutputs[1].at<float>(topIndex) * imgSrc.rows;
            float right = netOutputs[1].at<float>(rightIndex) * imgSrc.cols;
            float bottom = netOutputs[1].at<float>(bottomIndex) * imgSrc.rows;
            float width = right - left + 1;
            float height = bottom - top + 1;

            //Retrieve class label
            CGraphicsTextProperty textProperty;
            textProperty.m_color = {m_colors[classId+1][0], m_colors[classId+1][1], m_colors[classId+1][2]};
            textProperty.m_fontSize = 8;
            std::string className = classId < m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);
            std::string label = className + " : " + std::to_string(confidence);
            pGraphicsOutput->addText(label, left + 5, top + 5, textProperty);

            //Create rectangle graphics of bbox
            CGraphicsRectProperty rectProperty;
            rectProperty.m_category = className;
            rectProperty.m_penColor = {m_colors[classId+1][0], m_colors[classId+1][1], m_colors[classId+1][2]};
            auto graphicsObj = pGraphicsOutput->addRectangle(left, top, width, height, rectProperty);

            //Extract mask
            cv::Mat objMask(netOutputs[0].size[2], netOutputs[0].size[3], CV_32F, netOutputs[0].ptr<float>(n, classId));
            //Resize to the size of the box
            cv::resize(objMask, objMask, cv::Size(width, height), cv::INTER_LINEAR);
            //Apply thresholding to get the pixel wise mask
            cv::Mat objMaskBinary = (objMask > pParam->m_maskThreshold);
            objMaskBinary.convertTo(objMaskBinary, CV_8U);
            //Create label mask according to the object class (we do classId + 1 because 0 is the background label)
            cv::Mat classLabelImg(height, width, CV_8UC1, cv::Scalar(classId + 1));
            cv::Mat objClassMask(height, width, CV_8UC1, cv::Scalar(0));
            classLabelImg.copyTo(objClassMask, objMaskBinary);
            //Merge object label mask to final label image
            cv::Mat roi = labelImg(cv::Rect(left, top, width, height));
            cv::bitwise_or(roi, objClassMask, roi);

            //Store object values in result table
            std::vector<CObjectMeasure> results;
            results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidence, graphicsObj->getId(), className));
            results.emplace_back(CObjectMeasure(CMeasure::Id::BBOX, {left, top, width, height}, graphicsObj->getId(), className));
            pMeasureOutput->addObjectMeasures(results);
        }
    }

    auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if(pOutput)
        pOutput->setImage(labelImg);
}

void CMaskRCNN::generateColors()
{
    //The label value 0 is reserved for background pixels
    m_colors.push_back(cv::Vec3b(0, 0, 0));
    //Random colors then
    for(size_t i=1; i<m_classNames.size(); ++i)
    {
        cv::Vec3b color;
        for(int j=0; j<3; ++j)
            color[j] = (uchar)((double)std::rand() / (double)RAND_MAX * 255.0);

        m_colors.push_back(color);
    }
}
