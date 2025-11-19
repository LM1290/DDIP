import SwiftUI

// MARK: - Data Models
struct DrugResponse: Codable {
    let drugs: [String]
}

struct PredictionResponse: Codable {
    let drug_a: String
    let drug_b: String
    let probability: Double
    let risk_level: String
}

// MARK: - Main View
struct ContentView: View {
    // State
    @State private var drugs: [String] = []
    @State private var selectedDrugA = ""
    @State private var selectedDrugB = ""
    @State private var prediction: PredictionResponse?
    @State private var isLoading = false
    @State private var errorMessage = ""
    
    // Configuration
    // NOTE: If running on Simulator, use localhost.
    // If running on physical iPhone, use your computer's local IP (e.g., http://192.168.1.5:8000)
    let baseURL = "http://127.0.0.1:8000"

    var body: some View {
        ZStack {
            // 1. Modern Background
            LinearGradient(gradient: Gradient(colors: [Color(hex: "fdfbfb"), Color(hex: "ebedee")]), startPoint: .top, endPoint: .bottom)
                .edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 25) {
                
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "cross.case.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)
                    Text("BioGuard AI")
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundColor(.black)
                    Text("DDI Interaction Predictor")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .padding(.top, 40)
                
                // Drug Selection Cards
                VStack(spacing: 15) {
                    DrugPickerView(title: "First Compound", selection: $selectedDrugA, options: drugs)
                    DrugPickerView(title: "Second Compound", selection: $selectedDrugB, options: drugs)
                }
                .padding(.horizontal)
                
                // Analyze Button
                Button(action: fetchPrediction) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Text("Analyze Interaction")
                                .fontWeight(.bold)
                            Image(systemName: "waveform.path.ecg")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(LinearGradient(gradient: Gradient(colors: [Color.blue, Color.purple]), startPoint: .leading, endPoint: .trailing))
                    .foregroundColor(.white)
                    .cornerRadius(15)
                    .shadow(color: .blue.opacity(0.3), radius: 10, x: 0, y: 10)
                }
                .padding(.horizontal)
                .disabled(isLoading || drugs.isEmpty)

                // Result Card
                if let result = prediction {
                    ResultCard(result: result)
                        .transition(.scale.combined(with: .opacity))
                }
                
                Spacer()
            }
        }
        .onAppear(perform: fetchDrugs)
    }
    
    // MARK: - Networking
    func fetchDrugs() {
        guard let url = URL(string: "\(baseURL)/drugs") else { return }
        
        URLSession.shared.dataTask(with: url) { data, _, _ in
            if let data = data, let response = try? JSONDecoder().decode(DrugResponse.self, from: data) {
                DispatchQueue.main.async {
                    self.drugs = response.drugs
                    self.selectedDrugA = response.drugs.first ?? ""
                    self.selectedDrugB = response.drugs.last ?? ""
                }
            }
        }.resume()
    }
    
    func fetchPrediction() {
        guard let url = URL(string: "\(baseURL)/predict") else { return }
        self.isLoading = true
        self.prediction = nil
        
        let body: [String: String] = ["drug_a_name": selectedDrugA, "drug_b_name": selectedDrugB]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        URLSession.shared.dataTask(with: request) { data, _, _ in
            DispatchQueue.main.async {
                self.isLoading = false
                if let data = data, let response = try? JSONDecoder().decode(PredictionResponse.self, from: data) {
                    withAnimation {
                        self.prediction = response
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Subviews
struct DrugPickerView: View {
    let title: String
    @Binding var selection: String
    let options: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title.uppercased())
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(.gray)
                .padding(.leading, 5)
            
            HStack {
                Image(systemName: "pills.fill")
                    .foregroundColor(.gray)
                Picker(title, selection: $selection) {
                    ForEach(options, id: \.self) { option in
                        Text(option).tag(option)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .accentColor(.primary)
                Spacer()
            }
            .padding()
            .background(Color.white)
            .cornerRadius(12)
            .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
        }
    }
}

struct ResultCard: View {
    let result: PredictionResponse
    
    var isHighRisk: Bool {
        return result.probability > 0.5
    }
    
    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Prediction Result")
                    .font(.headline)
                    .foregroundColor(.gray)
                Spacer()
                Text(isHighRisk ? "CAUTION" : "SAFE")
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(6)
                    .background(isHighRisk ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .foregroundColor(isHighRisk ? .red : .green)
                    .cornerRadius(8)
            }
            
            Divider()
            
            HStack(alignment: .bottom) {
                Text("\(Int(result.probability * 100))%")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundColor(isHighRisk ? .red : .green)
                
                Text("Interaction\nProbability")
                    .font(.caption)
                    .foregroundColor(.gray)
                    .padding(.bottom, 8)
            }
            
            Text(isHighRisk ? "Adverse interaction detected between \(result.drug_a) and \(result.drug_b)." : "No significant interaction expected.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.top, 5)
        }
        .padding(20)
        .background(Color.white)
        .cornerRadius(20)
        .shadow(color: isHighRisk ? Color.red.opacity(0.2) : Color.green.opacity(0.2), radius: 15, x: 0, y: 5)
        .padding(.horizontal)
    }
}

// Helper for Hex Colors
extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)
        
        let r = (rgbValue & 0xff0000) >> 16
        let g = (rgbValue & 0xff00) >> 8
        let b = rgbValue & 0xff
        
        self.init(
            .sRGB,
            red: Double(r) / 0x255,
            green: Double(g) / 0x255,
            blue: Double(b) / 0x255,
            opacity: 1
        )
    }
}
