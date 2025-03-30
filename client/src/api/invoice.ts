import axios from 'axios';

const apiUrl = import.meta.env.VITE_SERVER_URL;

interface FetchInvoicesParams {
    page?: number;
    limit?: number;
}

interface Invoice {
    id: string;
    vendor_name: string;
    amount: string;
    due_date: string;
    description: string;
    paid: boolean;
    user_id: string;
}

interface FetchInvoicesResponse {
    total: number;
    page: number;
    limit: number;
    data: Invoice[];
}

export const fetchInvoices = async ({
                                        page = 1,
                                        limit = 10,
                                    }: FetchInvoicesParams = {}): Promise<FetchInvoicesResponse> => {
    const token = localStorage.getItem('token'); // Get the token from localStorage

    const response = await axios.get(`${apiUrl}/invoices`, {
        params: {
            page,
            limit,
        },
        headers: {
            Authorization: `Bearer ${token}`, // Include the token in the request headers
        },
    });
    return response.data;
};
