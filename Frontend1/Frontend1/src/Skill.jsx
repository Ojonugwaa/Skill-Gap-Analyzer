// import React, { useState } from "react";
// //eslint-disable-next-line no-unused-vars
// import { motion } from "framer-motion";
// import { FiUpload, FiCheckCircle } from "react-icons/fi";
// import { useNavigate } from "react-router-dom";
// import jsPDF from "jspdf";
// import html2canvas from "html2canvas";

// function Skill() {
//   const [jobTitle, setJobTitle] = useState("");
//   const [cvSkills, setCvSkills] = useState("");
//   const [file, setFile] = useState(null);
//   const [uploadSuccess, setUploadSuccess] = useState(false);
//   const [results, setResults] = useState(null);
//   const [error, setError] = useState("");
//   const [isLoading, setIsLoading] = useState(false);

//   const navigate = useNavigate();

//   const handleFileChange = (e) => {
//     const selectedFile = e.target.files[0];
//     if (
//       selectedFile &&
//       (selectedFile.type === "application/pdf" ||
//         selectedFile.type.includes("word"))
//     ) {
//       setFile(selectedFile);
//       setUploadSuccess(true);
//       setTimeout(() => setUploadSuccess(false), 3000);
//       setJobTitle("");
//       setCvSkills("");
//     } else {
//       setError("Please upload a PDF or DOCX file.");
//     }
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setError("");
//     setResults(null);
//     setIsLoading(true);

//     try {
//       const controller = new AbortController();
//       const timeoutId = setTimeout(() => {
//         controller.abort();
//       }, 600000);

//       const formData = new FormData();
//       if (jobTitle) formData.append("job_title", jobTitle);
//       if (cvSkills.trim()) formData.append("cv_skills", cvSkills);
//       if (file) formData.append("cv_file", file);

//       const response = await fetch("http://localhost:8000/analyze_skills", {
//         method: "POST",
//         body: formData,
//         signal: controller.signal,
//       });

//       clearTimeout(timeoutId);

//       if (!response.ok) {
//         const errorText = await response.text();
//         throw new Error(errorText || "Failed to analyze skills");
//       }

//       const data = await response.json();
//       setResults(data);
//     } catch (err) {
//       if (err.name === "AbortError") {
//         setError("⏱️ Request timed out. Try a smaller CV or use manual input.");
//       } else {
//         setError(err.message || "❌ Failed to analyze skills.");
//       }
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   // Function to handle PDF generation
//   const handleSaveAsPDF = () => {
//     const resultElement = document.getElementById("results-section");
//     if (!resultElement) {
//       alert("No results to save!");
//       return;
//     }

//     html2canvas(resultElement).then((canvas) => {
//       const imgData = canvas.toDataURL("image/png");
//       const pdf = new jsPDF({
//         orientation: "portrait",
//         unit: "mm",
//         format: "a4",
//       });
//       const pageWidth = pdf.internal.pageSize.getWidth();
//       const pageHeight = pdf.internal.pageSize.getHeight();
//       const imgProps = pdf.getImageProperties(imgData);
//       const imgWidth = pageWidth;
//       const imgHeight = (imgProps.height * pageWidth) / imgProps.width;

//       // Add image to PDF
//       pdf.addImage(imgData, "PNG", 0, 0, imgWidth, imgHeight);
//       pdf.save("PSSUQ_Results.pdf");
//     });
//   };

//   return (
//     <motion.div
//       initial={{ opacity: 0 }}
//       animate={{ opacity: 1 }}
//       transition={{ duration: 1 }}
//       className="container mx-auto px-4 py-8 max-w-4xl"
//     >
//       <motion.div
//         whileHover={{ scale: 1.01 }}
//         className="bg-white rounded-lg shadow-md p-6 mb-8"
//       >
//         <form onSubmit={handleSubmit} className="space-y-4">
//           <div className="space-y-2">
//             <label className="block text-gray-700 font-medium mb-2">
//               Upload CV (Optional)
//             </label>
//             <div className="flex items-center gap-4">
//               <label className="flex-1 cursor-pointer">
//                 <div
//                   className={`flex items-center justify-center px-4 py-2 border-2 border-dashed rounded-lg ${
//                     uploadSuccess
//                       ? "border-green-500 bg-green-50"
//                       : "border-gray-300 hover:border-indigo-500"
//                   }`}
//                 >
//                   <div className="flex items-center gap-2">
//                     <FiUpload className="text-gray-500" />
//                     <span className="text-gray-600">
//                       {file ? file.name : "Click to upload PDF/DOCX"}
//                     </span>
//                   </div>
//                   <input
//                     type="file"
//                     onChange={handleFileChange}
//                     accept=".pdf,.docx"
//                     className="hidden"
//                   />
//                 </div>
//               </label>
//               {uploadSuccess && (
//                 <motion.div
//                   initial={{ scale: 0 }}
//                   animate={{ scale: 1 }}
//                   className="text-green-600 flex items-center gap-1"
//                 >
//                   <FiCheckCircle />
//                   <span>Uploaded!</span>
//                 </motion.div>
//               )}
//             </div>
//           </div>

//           {!file && (
//             <>
//               <div>
//                 <label className="block text-gray-700 font-medium mb-2">
//                   Job Title
//                 </label>
//                 <input
//                   type="text"
//                   value={jobTitle}
//                   onChange={(e) => setJobTitle(e.target.value)}
//                   required={!file}
//                   className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
//                   placeholder="e.g. Data Analyst"
//                 />
//               </div>
//               <div>
//                 <label className="block text-gray-700 font-medium mb-2">
//                   Your Skills (comma-separated)
//                 </label>
//                 <textarea
//                   rows="4"
//                   value={cvSkills}
//                   onChange={(e) => setCvSkills(e.target.value)}
//                   required={!file}
//                   className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
//                   placeholder="e.g. Python, SQL, Excel, PowerBI"
//                 />
//               </div>
//             </>
//           )}

//           <motion.button
//             type="submit"
//             whileTap={{ scale: 0.95 }}
//             disabled={isLoading}
//             className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 ${
//               isLoading ? "opacity-70 cursor-not-allowed" : ""
//             }`}
//           >
//             {isLoading ? (
//               <span className="flex items-center justify-center">
//                 <svg
//                   className="animate-spin h-5 w-5 mr-2 text-white"
//                   viewBox="0 0 24 24"
//                 >
//                   <circle
//                     className="opacity-25"
//                     cx="12"
//                     cy="12"
//                     r="10"
//                     stroke="currentColor"
//                     strokeWidth="4"
//                   />
//                   <path
//                     className="opacity-75"
//                     fill="currentColor"
//                     d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
//                   />
//                 </svg>
//                 Analyzing...
//               </span>
//             ) : (
//               "Analyze Skills"
//             )}
//           </motion.button>

//           <div className="w-full flex justify-end">
//             <button
//               onClick={() => navigate("/curriculum")}
//               className="text-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200"
//             >
//               Curriculum Analyzer
//             </button>
//           </div>
//         </form>

//         {error && (
//           <motion.div
//             initial={{ opacity: 0, y: -10 }}
//             animate={{ opacity: 1, y: 0 }}
//             className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700"
//           >
//             <p>{error}</p>
//           </motion.div>
//         )}
//       </motion.div>

//       {results && (
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           className="bg-white rounded-lg shadow-md p-6"
//         >
//           <h3 className="text-xl font-bold text-gray-800 mb-4">
//             Results for: {results.job_title}
//           </h3>
//           <p className="mb-4">
//             <strong>Extracted Skills:</strong>{" "}
//             {results.extracted_skills?.length > 0
//               ? results.extracted_skills.join(", ")
//               : "None"}
//           </p>
//           <p className="mb-4">
//             <strong>Missing Skills:</strong>{" "}
//             {results.missing_skills?.length > 0
//               ? results.missing_skills.join(", ")
//               : "None 🎉"}
//           </p>
//           {results.course_recommendations &&
//             Object.keys(results.course_recommendations).length > 0 && (
//               <>
//                 <h4 className="text-lg font-medium mb-2">
//                   Course Recommendations
//                 </h4>
//                 {Object.entries(results.course_recommendations).map(
//                   ([skill, platforms]) => (
//                     <div key={skill} className="mb-4">
//                       <h5 className="font-semibold">{skill}</h5>
//                       <ul className="list-disc ml-6">
//                         {platforms?.coursera?.map((course, idx) => (
//                           <li key={`coursera-${idx}`}>
//                             Coursera:{" "}
//                             <a
//                               href={course.course_url}
//                               target="_blank"
//                               rel="noopener noreferrer"
//                               className="text-indigo-600 hover:underline"
//                             >
//                               {course.Title}
//                             </a>
//                           </li>
//                         ))}
//                         {platforms?.udemy?.map((course, idx) => (
//                           <li key={`udemy-${idx}`}>
//                             Udemy:{" "}
//                             <a
//                               href={course.course_url}
//                               target="_blank"
//                               rel="noopener noreferrer"
//                               className="text-indigo-600 hover:underline"
//                             >
//                               {course.Title}
//                             </a>
//                           </li>
//                         ))}
//                       </ul>
//                     </div>
//                   )
//                 )}
//               </>
//             )}
//           {/* Save as PDF Button */}
//           <button
//             onClick={handleSaveAsPDF}
//             className="mt-4 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg"
//           >
//             Save as PDF
//           </button>
//         </motion.div>
//       )}
//     </motion.div>
//   );
// }

// export default Skill;


import React, { useState } from "react";
//eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { FiUpload, FiCheckCircle } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import jsPDF from "jspdf";

// Updated SaveButton Component


const SaveButton = ({ results }) => {
  const handleSaveAsPDF = () => {
    const doc = new jsPDF();
    const marginLeft = 15;
    const pageWidth = doc.internal.pageSize.getWidth();
    const maxLineWidth = pageWidth - marginLeft * 2;

    let y = 20;

    doc.setFontSize(18);
    doc.text("Skill Analysis Results", marginLeft, y);
    y += 10;

    if (results.job_title) {
      doc.setFontSize(14);
      doc.text(`Job Title:`, marginLeft, y);
      doc.setFontSize(12);
      y += 7;
      doc.text(results.job_title, marginLeft, y, { maxWidth: maxLineWidth });
      y += 10;
    }

    if (results.extracted_skills) {
      doc.setFontSize(14);
      doc.text("Extracted Skills:", marginLeft, y);
      y += 7;

      const extractedSkillsText = results.extracted_skills.length > 0
        ? results.extracted_skills.join(", ")
        : "None";

      doc.setFontSize(12);
      doc.text(extractedSkillsText, marginLeft, y, { maxWidth: maxLineWidth });
      y += 10;
    }

    if (results.missing_skills) {
      doc.setFontSize(14);
      doc.text("Missing Skills:", marginLeft, y);
      y += 7;

      const missingSkillsText = results.missing_skills.length > 0
        ? results.missing_skills.join(", ")
        : "None 🎉";

      doc.setFontSize(12);
      doc.text(missingSkillsText, marginLeft, y, { maxWidth: maxLineWidth });
      y += 10;
    }

    if (results.course_recommendations && Object.keys(results.course_recommendations).length > 0) {
      doc.setFontSize(14);
      doc.text("Course Recommendations:", marginLeft, y);
      y += 7;

      doc.setFontSize(12);

      Object.entries(results.course_recommendations).forEach(([skill, platforms]) => {
        doc.text(`• ${skill}:`, marginLeft, y);
        y += 7;

        if (platforms.coursera) {
          platforms.coursera.forEach(course => {
            const line = `  Coursera: ${course.Title}`;
            doc.text(line, marginLeft + 8, y, { maxWidth: maxLineWidth - 8 });
            y += 7;
            // Add URL as a smaller font or separate line
            doc.setTextColor("blue");
            doc.textWithLink(course.course_url, marginLeft + 8, y, { url: course.course_url });
            doc.setTextColor("black");
            y += 7;
          });
        }

        if (platforms.udemy) {
          platforms.udemy.forEach(course => {
            const line = `  Udemy: ${course.Title}`;
            doc.text(line, marginLeft + 8, y, { maxWidth: maxLineWidth - 8 });
            y += 7;
            doc.setTextColor("blue");
            doc.textWithLink(course.course_url, marginLeft + 8, y, { url: course.course_url });
            doc.setTextColor("black");
            y += 7;
          });
        }

        y += 5;
        if (y > doc.internal.pageSize.getHeight() - 20) {
          doc.addPage();
          y = 20;
        }
      });
    }

    doc.save("Skill_Analysis_Results.pdf");
  };

  return (
    <button
      onClick={handleSaveAsPDF}
      className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg"
    >
      Save as PDF
    </button>
  );
};


function Skill() {
  const [jobTitle, setJobTitle] = useState("");
  const [cvSkills, setCvSkills] = useState("");
  const [file, setFile] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (
      selectedFile &&
      (selectedFile.type === "application/pdf" || selectedFile.type.includes("word"))
    ) {
      setFile(selectedFile);
      setUploadSuccess(true);
      setTimeout(() => setUploadSuccess(false), 3000);
      setJobTitle("");
      setCvSkills("");
      setError("");
    } else {
      setError("Please upload a PDF or DOCX file.");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults(null);
    setIsLoading(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
      }, 600000); // 10 minutes

      const formData = new FormData();
      if (jobTitle) formData.append("job_title", jobTitle);
      if (cvSkills.trim()) formData.append("cv_skills", cvSkills);
      if (file) formData.append("cv_file", file);

      const response = await fetch("http://localhost:8000/analyze_skills", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to analyze skills");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      if (err.name === "AbortError") {
        setError("⏱️ Request timed out. Try a smaller CV or use manual input.");
      } else {
        setError(err.message || "❌ Failed to analyze skills.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="container mx-auto px-4 py-8 max-w-4xl"
    >
      <motion.div whileHover={{ scale: 1.01 }} className="bg-white rounded-lg shadow-md p-6 mb-8">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="block text-gray-700 font-medium mb-2">Upload CV (Optional)</label>
            <div className="flex items-center gap-4">
              <label className="flex-1 cursor-pointer">
                <div
                  className={`flex items-center justify-center px-4 py-2 border-2 border-dashed rounded-lg ${
                    uploadSuccess ? "border-green-500 bg-green-50" : "border-gray-300 hover:border-indigo-500"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <FiUpload className="text-gray-500" />
                    <span className="text-gray-600">{file ? file.name : "Click to upload PDF/DOCX"}</span>
                  </div>
                  <input type="file" onChange={handleFileChange} accept=".pdf,.docx" className="hidden" />
                </div>
              </label>
              {uploadSuccess && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="text-green-600 flex items-center gap-1"
                >
                  <FiCheckCircle />
                  <span>Uploaded!</span>
                </motion.div>
              )}
            </div>
          </div>

          {!file && (
            <>
              <div>
                <label className="block text-gray-700 font-medium mb-2">Job Title</label>
                <input
                  type="text"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  required={!file}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g. Data Analyst"
                />
              </div>
              <div>
                <label className="block text-gray-700 font-medium mb-2">Your Skills (comma-separated)</label>
                <textarea
                  rows="4"
                  value={cvSkills}
                  onChange={(e) => setCvSkills(e.target.value)}
                  required={!file}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g. Python, SQL, Excel, PowerBI"
                />
              </div>
            </>
          )}

          <motion.button
            type="submit"
            whileTap={{ scale: 0.95 }}
            disabled={isLoading}
            className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 ${
              isLoading ? "opacity-70 cursor-not-allowed" : ""
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Analyzing...
              </span>
            ) : (
              "Analyze Skills"
            )}
          </motion.button>

          <div className="w-full flex justify-end">
            <button
              onClick={() => navigate("/curriculum")}
              className="text-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200"
            >
              Curriculum Analyzer
            </button>
          </div>
        </form>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700"
          >
            <p>{error}</p>
          </motion.div>
        )}
      </motion.div>

      {results && (
        <motion.div
          id="results-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4">Results for: {results.job_title}</h3>
          <p className="mb-4">
            <strong>Extracted Skills:</strong>{" "}
            {results.extracted_skills?.length > 0 ? results.extracted_skills.join(", ") : "None"}
          </p>
          <p className="mb-4">
            <strong>Missing Skills:</strong>{" "}
            {results.missing_skills?.length > 0 ? results.missing_skills.join(", ") : "None 🎉"}
          </p>
          {results.course_recommendations && Object.keys(results.course_recommendations).length > 0 && (
            <>
              <h4 className="text-lg font-medium mb-2">Course Recommendations</h4>
              {Object.entries(results.course_recommendations).map(([skill, platforms]) => (
                <div key={skill} className="mb-4">
                  <h5 className="font-semibold">{skill}</h5>
                  <ul className="list-disc ml-6">
                    {platforms?.coursera?.map((course, idx) => (
                      <li key={`coursera-${idx}`}>
                        Coursera:{" "}
                        <a
                          href={course.course_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-indigo-600 hover:underline"
                        >
                          {course.Title}
                        </a>
                      </li>
                    ))}
                    {platforms?.udemy?.map((course, idx) => (
                      <li key={`udemy-${idx}`}>
                        Udemy:{" "}
                        <a
                          href={course.course_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-indigo-600 hover:underline"
                        >
                          {course.Title}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </>
          )}
          {/* Integrated SaveButton */}
          <SaveButton results={results} resultElementId="results-section" fileName="Skill_Analysis_Results.pdf" />
        </motion.div>
      )}
    </motion.div>
  );
}

export default Skill;
