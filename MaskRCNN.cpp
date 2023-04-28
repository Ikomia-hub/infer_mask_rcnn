#include "MaskRCNN.h"
#include "IO/CInstanceSegIO.h"

CMaskRCNN::CMaskRCNN() : COcvDnnProcess(), CInstanceSegTask()
{
    m_pParam = std::make_shared<CMaskRCNNParam>();
}

CMaskRCNN::CMaskRCNN(const std::string& name, const std::shared_ptr<CMaskRCNNParam> &pParam)
    : COcvDnnProcess(), CInstanceSegTask(name)
{
    m_pParam = std::make_shared<CMaskRCNNParam>(*pParam);
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

    if (pInput == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

    if(pInput == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    //Force model files path
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString();
    pParam->m_structureFile = pluginDir + "/Model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    pParam->m_modelFile = pluginDir + "/Model/frozen_inference_graph.pb";
    pParam->m_labelsFile = pluginDir + "/Model/coco_names.txt";

    if (!Utils::File::isFileExist(pParam->m_modelFile))
    {
        std::cout << "Downloading model..." << std::endl;
        std::string downloadUrl = Utils::Plugin::getModelHubUrl() + "/" + m_name + "/frozen_inference_graph.pb";
        download(downloadUrl, pParam->m_modelFile);
    }

    CMat imgSrc = pInput->getImage();
    std::vector<cv::Mat> netOutputs;
    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn(pParam);
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            readClassNames(pParam->m_labelsFile);
            pParam->m_bUpdate = false;
        }
        forward(imgSrc, netOutputs, pParam);
    }
    catch(std::exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    emit m_signalHandler->doProgress();
    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();
    endTaskRun();
}

void CMaskRCNN::manageOutput(std::vector<cv::Mat> &netOutputs)
{
    manageMaskRCNNOutput(netOutputs);   
}

void CMaskRCNN::manageMaskRCNNOutput(std::vector<cv::Mat> &netOutputs)
{
    auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

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

            //Extract mask
            cv::Mat objMask(netOutputs[0].size[2], netOutputs[0].size[3], CV_32F, netOutputs[0].ptr<float>(n, classId));
            //Resize to the size of the box
            cv::resize(objMask, objMask, cv::Size(width, height), cv::INTER_LINEAR);
            //Apply thresholding to get the pixel wise mask
            cv::Mat objMaskBinary = (objMask > pParam->m_maskThreshold);
            objMaskBinary.convertTo(objMaskBinary, CV_8U);
            cv::Mat mask(imgSrc.rows, imgSrc.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat roi(mask, cv::Rect(left, top, width, height));
            objMaskBinary.copyTo(roi);

            addObject(n, CInstanceSegmentation::ObjectType::THING, classId, confidence, left, top, width, height, mask);
        }
    }
}
