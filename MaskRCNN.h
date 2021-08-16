#ifndef MASKRCNN_H
#define MASKRCNN_H

#include "MaskRCNNGlobal.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//--------------------------//
//----- CMaskRCNNParam -----//
//--------------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNNParam: public COcvDnnProcessParam
{
    public:

        CMaskRCNNParam() : COcvDnnProcessParam()
        {
            m_framework = Framework::TENSORFLOW;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
            m_confidence = std::stod(paramMap.at("confidence"));
            m_maskThreshold = std::stod(paramMap.at("maskThreshold"));
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
            paramMap.insert(std::make_pair("maskThreshold", std::to_string(m_maskThreshold)));
            return paramMap;
        }

    public:

        double              m_confidence = 0.5;
        double              m_maskThreshold = 0.3;
};

//---------------------//
//----- CMaskRCNN -----//
//---------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNN: public COcvDnnProcess
{
    public:

        CMaskRCNN();
        CMaskRCNN(const std::string& name, const std::shared_ptr<CMaskRCNNParam>& pParam);

        size_t                  getProgressSteps() override;
        int                     getNetworkInputSize() const override;
        double                  getNetworkInputScaleFactor() const override;
        cv::Scalar              getNetworkInputMean() const override;
        std::vector<cv::String> getOutputsNames() const override;

        void                    run() override;

    private:

        void                    manageOutput(std::vector<cv::Mat> &netOutputs);
        void                    manageMaskRCNNOutput(std::vector<cv::Mat> &netOutputs);

        std::vector<cv::Vec3b>  generateColorMap(const cv::Mat &netOutput, bool bWithBackgroundClass);

    private:
        std::vector<cv::Vec3b>  m_colors;
};

//----------------------------//
//----- CMaskRCNNFactory -----//
//----------------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNNFactory : public CTaskFactory
{
    public:

        CMaskRCNNFactory()
        {
            m_info.m_name = QObject::tr("Mask RCNN").toStdString();
            m_info.m_shortDescription = QObject::tr("Semantic segmentation based on Faster R-CNN method").toStdString();
            m_info.m_description = QObject::tr("We present a conceptually simple, flexible, and general framework for object instance segmentation. "
                                               "Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. "
                                               "The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask "
                                               "in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and "
                                               "adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, "
                                               "e.g., allowing us to estimate human poses in the same framework. "
                                               "We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, "
                                               "bounding-box object detection, and person keypoint detection. "
                                               "Without tricks, Mask R-CNN outperforms all existing, single-model entries on every task, "
                                               "including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and "
                                               "help ease future research in instance-level recognition. Code will be made available.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Object/Segmentation").toStdString();
            m_info.m_version = "1.0.0";
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick";
            m_info.m_article = "Mask R-CNN";
            m_info.m_journal = "ICCV";
            m_info.m_year = 2017;
            m_info.m_license = "Apache 2 License";
            m_info.m_repo = "https://github.com/tensorflow/models/tree/master/research";
            m_info.m_keywords = "deep,learning,segmentation,semantic,tensorflow,Faster R-CNN";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CMaskRCNNParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CMaskRCNN>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CMaskRCNNParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CMaskRCNN>(m_info.m_name, paramPtr);
        }
};

//---------------------------//
//----- CMaskRCNNWidget -----//
//---------------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNNWidget: public COcvWidgetDnnCore
{
    public:

        CMaskRCNNWidget(QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            init();
        }
        CMaskRCNNWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(pParam, parent)
        {
            m_pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(pParam);
            init();
        }

    private:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CMaskRCNNParam>();

            auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
            assert(pParam);

            auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
            auto pSpinMaskThreshold = addDoubleSpin(tr("Mask threshold"), pParam->m_maskThreshold, 0.0, 1.0, 0.1, 2);

            //Connections
            connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
                assert(pParam);
                pParam->m_confidence = val;
            });
            connect(pSpinMaskThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CMaskRCNNParam>(m_pParam);
                assert(pParam);
                pParam->m_maskThreshold = val;
            });
        }

        void onApply() override
        {
            emit doApplyProcess(m_pParam);
        }
};

//----------------------------------//
//----- CMaskRCNNWidgetFactory -----//
//----------------------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNNWidgetFactory : public CWidgetFactory
{
    public:

        CMaskRCNNWidgetFactory()
        {
            m_name = QObject::tr("Mask RCNN").toStdString();
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CMaskRCNNWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class MASKRCNNSHARED_EXPORT CMaskRCNNInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CMaskRCNNFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CMaskRCNNWidgetFactory>();
        }
};

#endif // MASKRCNN_H
